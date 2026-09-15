"""Metadata-only timing for provider requests; never inspect bodies or headers."""

import asyncio
import logging
import time
from contextlib import suppress
from contextvars import ContextVar
from uuid import uuid4

logger = logging.getLogger("gateway.requests")
logger.setLevel(logging.INFO)
_request: ContextVar[dict | None] = ContextVar("gateway_request_timing", default=None)


def log_request_stage(stage: str, **fields) -> None:
    context = _request.get()
    if context is None:
        return
    now = time.monotonic()
    previous_stage = context["stage"]
    stage_elapsed = now - context["stage_started"]
    if stage != "waiting":
        context["stage"] = stage
        context["stage_started"] = now
    logger.info(
        "gateway_request session=%s request=%s event=%s elapsed_s=%.3f "
        "previous_stage=%s stage_elapsed_s=%.3f details=%s",
        context["session"],
        context["id"],
        stage,
        now - context["started"],
        previous_stage,
        stage_elapsed,
        fields,
    )


class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith("/sessions/"):
            return await self.app(scope, receive, send)
        started = time.monotonic()
        context = {
            "id": uuid4().hex,
            "session": path.split("/")[2],
            "started": started,
            "stage": "received",
            "stage_started": started,
        }
        token = _request.set(context)
        log_request_stage("received", method=scope.get("method"), path=path)
        status = None
        complete = False
        first_body = True

        async def heartbeat():
            while True:
                await asyncio.sleep(60)
                log_request_stage("waiting", active_stage=context["stage"])

        async def traced_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                log_request_stage("client_disconnected")
            elif message["type"] == "http.request" and not message.get("more_body", False):
                log_request_stage("request_body_complete")
            return message

        async def traced_send(message):
            nonlocal status, complete, first_body
            if message["type"] == "http.response.start":
                route = scope.get("route")
                log_request_stage(
                    "response_headers_sending",
                    status=message["status"],
                    route=getattr(route, "path", None),
                    not_found_reason=("route_not_found" if route is None else "handler_not_found")
                    if message["status"] == 404
                    else None,
                )
            elif message["type"] == "http.response.body" and first_body:
                log_request_stage("response_first_body_sending")
            await send(message)
            if message["type"] == "http.response.start":
                status = message["status"]
                log_request_stage("response_headers", status=status)
            elif message["type"] == "http.response.body":
                if first_body:
                    log_request_stage("response_first_body")
                    first_body = False
                if not message.get("more_body", False):
                    complete = True
                    log_request_stage("response_complete", status=status)

        waiter = asyncio.create_task(heartbeat())
        try:
            await self.app(scope, traced_receive, traced_send)
        except asyncio.CancelledError:
            log_request_stage("cancelled")
            raise
        except Exception as exc:
            log_request_stage("error", error_type=type(exc).__name__)
            raise
        finally:
            waiter.cancel()
            with suppress(asyncio.CancelledError):
                await waiter
            log_request_stage("request_end", status=status, response_complete=complete)
            _request.reset(token)
