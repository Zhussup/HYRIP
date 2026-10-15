"""SSE framing for tutor streams: `data:` frames, an error frame, idle heartbeats."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterable

from ..errors import HyripError

HEARTBEAT_SECONDS = 15.0


def data_frame(payload: str) -> str:
    return f"data: {json.dumps({'token': payload})}\n\n"


def error_frame(message: str) -> str:
    return f"data: {json.dumps({'error': message})}\n\n"


def done_frame() -> str:
    return 'data: {"done": true}\n\n'


async def stream_sse(chunks_factory) -> AsyncIterator[str]:
    """Pump a sync chunk iterator from a worker thread into SSE frames.

    Any error becomes a single SSE error frame followed by `done`; the client
    sees a heartbeat comment whenever nothing arrives for 15 seconds.
    """
    queue: asyncio.Queue[object] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def work() -> None:
        put = loop.call_soon_threadsafe
        try:
            for chunk in chunks_factory():
                put(queue.put_nowait, chunk)
        except Exception as exc:
            put(queue.put_nowait, exc if isinstance(exc, Exception) else RuntimeError(exc))
        finally:
            put(queue.put_nowait, None)

    threading.Thread(target=work, daemon=True, name="hyrip-sse-pump").start()
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
        except TimeoutError:
            yield ": heartbeat\n\n"
            continue
        if item is None:
            yield done_frame()
            return
        if isinstance(item, Exception):
            if isinstance(item, HyripError):
                yield error_frame(item.user_message())
            else:
                yield error_frame(str(item))
            yield done_frame()
            return
        yield data_frame(item)


def sse_headers() -> dict[str, str]:
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def all_text(chunks: Iterable[str]) -> str:
    return "".join(chunks)
