"""
Scribe MCP connector — мост между Claude и ElevenLabs Speech-to-Text.

Зачем он нужен:
Облачные сессии Claude не имеют прямого доступа в интернет, но коннекторы
(MCP) под это ограничение не попадают. Этот сервер живёт на Render, ходит
в ElevenLabs сам и отдаёт Claude готовую расшифровку.

Ключевая идея — асинхронность. Запись на 90 минут расшифровывается
несколько минут; держать всё это время HTTP-соединение нельзя.
Поэтому submit_transcription ставит задачу и сразу возвращает id,
а get_transcription забирает результат отдельным вызовом.

Побочный плюс: сервер может спокойно засыпать между вызовами
(бесплатный тариф Render это делает через 15 минут простоя) —
результат хранится на стороне ElevenLabs и никуда не денется.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route

# --------------------------------------------------------------------------
# Конфигурация — всё через переменные окружения Render, ничего в коде
# --------------------------------------------------------------------------

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
CONNECTOR_TOKEN = os.environ.get("CONNECTOR_TOKEN", "")

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = os.environ.get("SCRIBE_MODEL", "scribe_v2")

# Сколько речевых реплик отдавать за один вызов get_transcription.
# Полуторачасовое совещание — это тысячи реплик; отдавать их разом
# бессмысленно, Claude читает порциями.
DEFAULT_PAGE_SIZE = 150

if not ELEVENLABS_API_KEY:
    raise RuntimeError(
        "Не задана переменная окружения ELEVENLABS_API_KEY. "
        "Задайте её в настройках сервиса на Render."
    )

if not CONNECTOR_TOKEN or len(CONNECTOR_TOKEN) < 24:
    raise RuntimeError(
        "Не задана переменная окружения CONNECTOR_TOKEN, либо она короче 24 символов. "
        "Это секрет, который закрывает коннектор от посторонних — сгенерируйте длинную "
        "случайную строку."
    )

# Кэш результатов, пришедших через webhook. Не обязателен: основной способ
# получить расшифровку — опрос ElevenLabs. Webhook нужен как подстраховка
# и как способ не ждать, если опрос по какой-то причине не сработает.
_webhook_cache: dict[str, dict[str, Any]] = {}


def _headers() -> dict[str, str]:
    return {"xi-api-key": ELEVENLABS_API_KEY}


# --------------------------------------------------------------------------
# Разбор ответа ElevenLabs
# --------------------------------------------------------------------------


def _words_to_turns(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    ElevenLabs отдаёт расшифровку пословно, с отметкой говорящего у каждого
    слова. Для чтения это непригодно, поэтому склеиваем подряд идущие слова
    одного говорящего в реплику.
    """
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for w in words:
        # Паузы и знаки препинания приходят отдельными элементами —
        # они прилипают к текущей реплике, а не начинают новую.
        if w.get("type") == "spacing":
            if current is not None:
                current["text"] += w.get("text", " ")
            continue

        speaker = w.get("speaker_id") or "speaker_unknown"

        if current is None or current["speaker"] != speaker:
            if current is not None:
                current["text"] = current["text"].strip()
                turns.append(current)
            current = {
                "speaker": speaker,
                "start": w.get("start"),
                "end": w.get("end"),
                "text": w.get("text", ""),
            }
        else:
            current["text"] += w.get("text", "")
            current["end"] = w.get("end")

    if current is not None:
        current["text"] = current["text"].strip()
        turns.append(current)

    return [t for t in turns if t["text"]]


def _fmt_ts(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------
# Инструменты MCP
# --------------------------------------------------------------------------

mcp = FastMCP("scribe", stateless_http=True)


@mcp.tool()
async def submit_transcription(
    source_url: str,
    language_code: str | None = None,
    num_speakers: int | None = None,
    diarize: bool = True,
) -> dict[str, Any]:
    """Поставить аудио в очередь на расшифровку в ElevenLabs Scribe.

    Возвращается сразу, не дожидаясь результата — расшифровка длинной записи
    занимает минуты. Полученный request_id передайте в get_transcription.

    Args:
        source_url: Прямая ссылка на аудио- или видеофайл. ElevenLabs скачает
            его сам, поэтому ссылка должна быть публично доступной (подойдёт
            временная подписанная ссылка, например из Plaud). Лимит — 2 ГБ.
        language_code: Код языка (ISO-639-1/639-3), например "kk" или "ru".
            Если не указан, язык определяется автоматически. Для записей,
            где говорят на двух языках вперемешку, лучше не указывать.
        num_speakers: Ожидаемое число говорящих, от 1 до 32. Подсказка
            улучшает разделение по ролям, но не обязательна.
        diarize: Размечать ли говорящих. По умолчанию да.
    """
    data: dict[str, Any] = {
        "model_id": DEFAULT_MODEL,
        "source_url": source_url,
        "diarize": str(diarize).lower(),
        "webhook": "true",
    }
    if language_code:
        data["language_code"] = language_code
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{ELEVENLABS_BASE}/speech-to-text",
            headers=_headers(),
            data=data,
        )

    if resp.status_code >= 400:
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": resp.text[:2000],
            "hint": (
                "401/403 — проверьте ключ и его права на Speech to Text. "
                "422 — скорее всего ElevenLabs не смог скачать файл по ссылке: "
                "проверьте, что она ещё не истекла."
            ),
        }

    body = resp.json()
    request_id = (
        body.get("request_id")
        or body.get("transcription_id")
        or body.get("id")
    )

    return {
        "ok": True,
        "request_id": request_id,
        "raw": body,
        "next": "Вызовите get_transcription с этим request_id через 1-3 минуты.",
    }


@mcp.tool()
async def get_transcription(
    request_id: str,
    offset: int = 0,
    limit: int = DEFAULT_PAGE_SIZE,
    as_text: bool = True,
) -> dict[str, Any]:
    """Забрать готовую расшифровку по идентификатору.

    Если расшифровка ещё не закончена, вернётся status="processing" —
    повторите вызов через минуту.

    Длинные записи отдаются порциями: смотрите поле has_more и вызывайте
    снова, увеличив offset.

    Args:
        request_id: Идентификатор из submit_transcription.
        offset: С какой реплики начать выдачу.
        limit: Сколько реплик вернуть за раз.
        as_text: Вернуть готовый текст с таймкодами и говорящими (по умолчанию)
            либо структурированный список реплик.
    """
    payload: dict[str, Any] | None = None

    # Сначала смотрим, не пришёл ли результат через webhook
    cached = _webhook_cache.get(request_id)
    if cached:
        payload = cached

    if payload is None:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                f"{ELEVENLABS_BASE}/speech-to-text/transcripts/{request_id}",
                headers=_headers(),
            )

        if resp.status_code in (404, 425):
            return {
                "ok": True,
                "status": "processing",
                "message": "Расшифровка ещё выполняется. Повторите через минуту.",
            }
        if resp.status_code >= 400:
            return {
                "ok": False,
                "status_code": resp.status_code,
                "error": resp.text[:2000],
            }
        payload = resp.json()

    words = payload.get("words") or []
    if not words and not payload.get("text"):
        return {
            "ok": True,
            "status": "processing",
            "message": "Расшифровка ещё выполняется. Повторите через минуту.",
        }

    turns = _words_to_turns(words)
    window = turns[offset : offset + limit]

    result: dict[str, Any] = {
        "ok": True,
        "status": "done",
        "language_code": payload.get("language_code"),
        "language_probability": payload.get("language_probability"),
        "total_turns": len(turns),
        "offset": offset,
        "returned": len(window),
        "has_more": offset + len(window) < len(turns),
    }

    if as_text:
        result["text"] = "\n".join(
            f"[{_fmt_ts(t['start'])}] {t['speaker']}: {t['text']}" for t in window
        )
    else:
        result["turns"] = window

    # Если разметка говорящих не сработала, отдаём сплошной текст —
    # лучше так, чем ничего.
    if not turns and payload.get("text"):
        result["text"] = payload["text"]
        result["note"] = "Разметка по говорящим недоступна, отдан сплошной текст."

    return result


@mcp.tool()
async def check_status() -> dict[str, Any]:
    """Проверить, что коннектор жив и ключ ElevenLabs принимается.

    Полезно вызвать сразу после подключения, чтобы убедиться, что всё
    настроено, не тратя минуты аудио.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{ELEVENLABS_BASE}/user", headers=_headers())

    if resp.status_code == 401:
        return {"ok": False, "error": "Ключ ElevenLabs отклонён (401)."}
    if resp.status_code == 403:
        return {
            "ok": True,
            "key": "принят",
            "note": (
                "У ключа нет прав на чтение профиля — это нормально, если ключ "
                "ограничен только Speech to Text. Расшифровке это не мешает."
            ),
        }
    if resp.status_code >= 400:
        return {"ok": False, "status_code": resp.status_code, "error": resp.text[:500]}

    body = resp.json()
    sub = body.get("subscription") or {}
    return {
        "ok": True,
        "key": "принят",
        "model": DEFAULT_MODEL,
        "tier": sub.get("tier"),
        "character_count": sub.get("character_count"),
        "character_limit": sub.get("character_limit"),
    }


# --------------------------------------------------------------------------
# HTTP-обвязка
# --------------------------------------------------------------------------


async def webhook_receiver(request: Request) -> JSONResponse:
    """Принимает готовые расшифровки от ElevenLabs (необязательный путь)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)

    if body.get("type") == "speech_to_text_transcription":
        data = body.get("data") or {}
        rid = data.get("request_id")
        transcription = data.get("transcription")
        if rid and transcription:
            transcription["_received_at"] = time.time()
            _webhook_cache[rid] = transcription
            # Кэш не должен расти бесконечно
            if len(_webhook_cache) > 200:
                oldest = sorted(
                    _webhook_cache.items(),
                    key=lambda kv: kv[1].get("_received_at", 0),
                )[:50]
                for k, _ in oldest:
                    _webhook_cache.pop(k, None)

    return JSONResponse({"ok": True})


async def healthz(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp.session_manager.run():
        yield


# Коннектор доступен по секретному пути. Без токена в URL сервер отдаёт 404,
# поэтому случайно наткнуться на него и потратить ваши минуты нельзя.
app = Starlette(
    routes=[
        Route("/healthz", healthz, methods=["GET"]),
        Route("/webhook", webhook_receiver, methods=["POST"]),
        Mount(f"/t/{CONNECTOR_TOKEN}", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)
