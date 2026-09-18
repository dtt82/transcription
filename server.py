"""
Scribe — коннектор расшифровки для Claude (ElevenLabs Speech-to-Text).

Аудио на этот сервер не скачивается: ссылка передаётся в ElevenLabs через
`source_url`, файл забирает сам ElevenLabs. Поэтому сервер лёгкий и спокойно
живёт на бесплатном тарифе Render.

Расшифровка асинхронная. `submit_transcription` ставит задачу и сразу отдаёт
`request_id`; `get_transcription` забирает готовый текст. Между вызовами сервер
может заснуть — состояние не теряется, потому что результат лежит у ElevenLabs
и запрашивается по `request_id`.

Переменные окружения:
  ELEVENLABS_API_KEY      — ключ ElevenLabs (обязательно)
  CONNECTOR_TOKEN         — секрет в пути: https://<host>/t/<CONNECTOR_TOKEN>/mcp
  SCRIBE_MODEL            — модель, по умолчанию scribe_v2
  PORT                    — порт HTTP, по умолчанию 8000
  ELEVENLABS_WEBHOOK_ID   — необязательно: конкретный webhook в аккаунте
  ELEVENLABS_WEBHOOK_SECRET — необязательно: проверка подписи входящих webhook
  ALLOWED_AUDIO_HOSTS     — через запятую; по умолчанию любые https-ссылки ("*")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

# ---------------------------------------------------------------- конфигурация

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
TOKEN = os.environ.get("CONNECTOR_TOKEN", "").strip()
MODEL = os.environ.get("SCRIBE_MODEL", "scribe_v2").strip() or "scribe_v2"
PORT = int(os.environ.get("PORT", "8000"))
EL_BASE = os.environ.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").rstrip("/")
WEBHOOK_ID = os.environ.get("ELEVENLABS_WEBHOOK_ID", "").strip()
WEBHOOK_SECRET = os.environ.get("ELEVENLABS_WEBHOOK_SECRET", "").strip()
ALLOWED_HOSTS = [
    h.strip().lower()
    for h in os.environ.get("ALLOWED_AUDIO_HOSTS", "*").split(",")
    if h.strip()
]

if not API_KEY:
    raise RuntimeError("Не задана переменная окружения ELEVENLABS_API_KEY")
if len(TOKEN) < 24:
    raise RuntimeError("CONNECTOR_TOKEN не задан или короче 24 символов")

CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "12000"))
CACHE_TTL = 12 * 3600

# Кэш результатов, пришедших через webhook. Не обязателен: без него текст
# забирается у ElevenLabs напрямую по request_id.
_cache: dict[str, dict] = {}


def _cache_put(request_id: str, data: dict) -> None:
    now = time.time()
    for k, v in list(_cache.items()):
        if now - v.get("at", now) > CACHE_TTL:
            _cache.pop(k, None)
    _cache[request_id] = {"at": now, "data": data}


# ------------------------------------------------------------------ утилиты

def _check_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme != "https":
        raise ValueError("Нужна прямая https-ссылка на аудио")
    if "*" in ALLOWED_HOSTS:
        return
    host = (p.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise ValueError(f"Домен {host} не разрешён (ALLOWED_AUDIO_HOSTS)")


def _ts(sec: float | None) -> str:
    sec = int(sec or 0)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _format_transcript(data: dict) -> str:
    """Реплики по говорящим, каждая с отметкой времени."""
    words = data.get("words") or []
    lang = data.get("language_code", "?")
    prob = data.get("language_probability")
    header = f"[язык: {lang}" + (f", уверенность {prob:.0%}" if prob else "") + "]\n"

    if not words or not any(w.get("speaker_id") for w in words):
        return header + (data.get("text") or "")

    lines: list[str] = []
    cur: str | None = None
    start = 0.0
    buf: list[str] = []
    for w in words:
        if w.get("type") == "audio_event":
            continue
        if w.get("type") == "spacing":
            buf.append(" ")
            continue
        sp = w.get("speaker_id") or cur
        if sp != cur and "".join(buf).strip():
            lines.append(f"[{_ts(start)}] {cur}: {''.join(buf).strip()}")
            buf = []
        if sp != cur or not buf:
            start = w.get("start") or start
        cur = sp
        buf.append(w.get("text", ""))
    if "".join(buf).strip():
        lines.append(f"[{_ts(start)}] {cur}: {''.join(buf).strip()}")
    return header + "\n".join(lines)


def _split(text: str, size: int) -> list[str]:
    """Режет транскрипт на части по границам строк."""
    parts: list[str] = []
    cur = ""
    for line in text.splitlines(keepends=True):
        if cur and len(cur) + len(line) > size:
            parts.append(cur)
            cur = ""
        cur += line
    if cur or not parts:
        parts.append(cur)
    return parts


def _net_error(e: Exception) -> str:
    return f"Не удалось связаться с ElevenLabs: {type(e).__name__}: {e}"[:400]


def _el_error(resp: httpx.Response) -> str:
    body = resp.text[:400]
    hints = {
        401: "ключ отозван или у него нет прав на Speech to Text",
        403: "ключу не хватает прав",
        422: "ElevenLabs не смог обработать запрос — чаще всего не скачалась ссылка на аудио (истекла)",
        429: "превышен лимит запросов, повторите позже",
    }
    hint = hints.get(resp.status_code)
    return f"Ошибка ElevenLabs {resp.status_code}" + (f" ({hint})" if hint else "") + f": {body}"


# -------------------------------------------------------------------- MCP

mcp = FastMCP(
    name="Scribe",
    instructions=(
        "Расшифровка аудио по https-ссылке через ElevenLabs Scribe "
        "(казахский kk, русский ru, английский en и 90+ языков) с разделением "
        "по говорящим. Для записей Plaud: возьмите presigned_url из Plaud "
        "get_file и передайте в submit_transcription — он вернёт request_id. "
        "Затем вызывайте get_transcription(request_id); если ответ 'ещё "
        "обрабатывается', повторите через минуту. Длинный текст отдаётся "
        "частями — забирайте следующие через параметр part."
    ),
    stateless_http=True,
    streamable_http_path=f"/t/{TOKEN}/mcp",
    host="0.0.0.0",
)

@mcp.tool()
async def submit_transcription(
    audio_url: str,
    language_code: str | None = None,
    diarize: bool = True,
    num_speakers: int | None = None,
    title: str | None = None,
) -> str:
    """Ставит аудио в очередь на расшифровку и возвращает request_id.

    Args:
        audio_url: Прямая https-ссылка на аудио или видео (для Plaud — presigned_url).
        language_code: 'kk', 'ru', 'en' и т.п. Не указан — автоопределение
            (обычно лучше для смешанной казахско-русской речи).
        diarize: Размечать говорящих. По умолчанию True.
        num_speakers: Подсказка по числу участников, 1–32. Необязательно.
        title: Название записи — попадёт в ответ, чтобы не перепутать задачи.
    """
    _check_url(audio_url)

    form: dict[str, str] = {
        "model_id": MODEL,
        "source_url": audio_url,
        "webhook": "true",
        "diarize": str(bool(diarize)).lower(),
        "tag_audio_events": "false",
        "timestamps_granularity": "word",
    }
    if language_code:
        form["language_code"] = language_code
    if num_speakers:
        form["num_speakers"] = str(int(num_speakers))
    if WEBHOOK_ID:
        form["webhook_id"] = WEBHOOK_ID
    if title:
        form["webhook_metadata"] = json.dumps({"title": title}, ensure_ascii=False)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30)) as c:
            resp = await c.post(
                f"{EL_BASE}/v1/speech-to-text",
                headers={"xi-api-key": API_KEY},
                files={k: (None, v) for k, v in form.items()},
            )
    except Exception as e:  # noqa: BLE001
        return _net_error(e)

    if resp.status_code >= 400:
        return _el_error(resp)

    data = resp.json()
    request_id = data.get("request_id") or data.get("transcription_id")

    # Если аккаунт отвечает синхронно (webhook не включён) — текст уже здесь.
    if not request_id:
        if data.get("text") or data.get("words"):
            return (f"«{title}»\n" if title else "") + _format_transcript(data)
        return f"Неожиданный ответ ElevenLabs: {json.dumps(data, ensure_ascii=False)[:500]}"

    return (
        f"Задача поставлена{f' — «{title}»' if title else ''}.\n"
        f"request_id: {request_id}\n"
        f"Заберите текст вызовом get_transcription(request_id='{request_id}') "
        f"— примерно через минуту на каждые 30 минут записи."
    )


@mcp.tool()
async def get_transcription(request_id: str, part: int = 1) -> str:
    """Забирает готовую расшифровку по request_id. Длинный текст — частями.

    Args:
        request_id: Идентификатор из submit_transcription.
        part: Номер части, начиная с 1.
    """
    request_id = request_id.strip()
    cached = _cache.get(request_id)
    if cached:
        data = cached["data"]
    else:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30)) as c:
                resp = await c.get(
                    f"{EL_BASE}/v1/speech-to-text/transcripts/{request_id}",
                    headers={"xi-api-key": API_KEY},
                )
        except Exception as e:  # noqa: BLE001
            return _net_error(e)
        if resp.status_code in (404, 409, 425):
            return (
                f"processing — расшифровка ещё идёт (request_id={request_id}). "
                "Повторите через минуту."
            )
        if resp.status_code >= 400:
            return _el_error(resp)
        data = resp.json()
        status = str(data.get("status") or "").lower()
        if status in ("processing", "queued", "pending", "in_progress"):
            return (
                f"processing — расшифровка ещё идёт (request_id={request_id}). "
                "Повторите через минуту."
            )
        if status in ("failed", "error"):
            return f"ElevenLabs не смог расшифровать запись: {json.dumps(data, ensure_ascii=False)[:400]}"
        _cache_put(request_id, data)

    text = _format_transcript(data)
    parts = _split(text, CHUNK_CHARS)
    idx = max(1, min(int(part), len(parts)))
    head = f"[часть {idx} из {len(parts)}]\n" if len(parts) > 1 else ""
    tail = (
        f"\n\n…продолжение: get_transcription(request_id='{request_id}', part={idx + 1})"
        if idx < len(parts)
        else ""
    )
    return head + parts[idx - 1] + tail


@mcp.tool()
async def check_status() -> str:
    """Проверяет, что ключ ElevenLabs принимается. Минуты аудио не расходуются."""
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(
                f"{EL_BASE}/v1/user/subscription", headers={"xi-api-key": API_KEY}
            )
    except Exception as e:  # noqa: BLE001
        return _net_error(e)
    if r.status_code >= 400:
        return "Ключ не принят. " + _el_error(r)
    d = r.json()
    used = d.get("character_count", 0)
    limit = d.get("character_limit", 0)
    return (
        f"Ключ принят. Тариф: {d.get('tier')}; использовано {used:,} из {limit:,} "
        f"кредитов (осталось {max(limit - used, 0):,}). Модель: {MODEL}."
    )


# --------------------------------------------------------- HTTP-обвязка

async def healthz(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _valid_signature(raw: bytes, header: str) -> bool:
    """Проверка подписи ElevenLabs: заголовок вида 't=...,v0=...'."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
        expected = hmac.new(
            WEBHOOK_SECRET.encode(),
            f"{parts['t']}.".encode() + raw,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, parts.get("v0", ""))
    except Exception:
        return False


async def webhook(request: Request) -> JSONResponse:
    """Необязательный приёмник результатов от ElevenLabs."""
    raw = await request.body()
    if WEBHOOK_SECRET:
        sig = request.headers.get("elevenlabs-signature", "")
        if not _valid_signature(raw, sig):
            return JSONResponse({"error": "bad signature"}, status_code=401)
    try:
        event = json.loads(raw)
    except Exception:
        return JSONResponse({"error": "bad json"}, status_code=400)

    data = event.get("data") or {}
    request_id = data.get("request_id")
    transcription = data.get("transcription")
    if request_id and transcription:
        _cache_put(request_id, transcription)
    return JSONResponse({"ok": True})


app = mcp.streamable_http_app()
app.router.routes.append(Route("/healthz", healthz, methods=["GET"]))
app.router.routes.append(Route("/webhook", webhook, methods=["POST"]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=False)
