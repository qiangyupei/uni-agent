import asyncio
import logging

import pytest

from uni_agent.gateway.request_logging import RequestLoggingMiddleware, _request, log_request_stage


@pytest.mark.parametrize("mode", ["complete", "disconnect", "error", "cancel"])
def test_request_logging_preserves_transport_and_cleans_context(mode, caplog):
    caplog.set_level(logging.INFO, logger="gateway.requests")
    sent = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def app(scope, receive, send):
        log_request_stage("backend_start", backend_request_id="session-test")
        if mode == "error":
            raise ValueError("private-error-content")
        if mode == "cancel":
            raise asyncio.CancelledError()
        if mode == "disconnect":
            assert (await receive())["type"] == "http.disconnect"
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"private-response", "more_body": False})

    async def run():
        try:
            await RequestLoggingMiddleware(app)(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/sessions/session-test/v1/messages",
                    "query_string": b"key=private-query",
                    "headers": [(b"authorization", b"private-token")],
                },
                receive,
                send,
            )
        finally:
            assert _request.get() is None
            assert len(asyncio.all_tasks()) == 1

    if mode in {"error", "cancel"}:
        with pytest.raises(ValueError if mode == "error" else asyncio.CancelledError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    event = {
        "complete": "response_complete",
        "disconnect": "client_disconnected",
        "error": "error",
        "cancel": "cancelled",
    }
    assert f"event={event[mode]}" in caplog.text
    assert "event=request_end" in caplog.text
    assert "session=session-test" in caplog.text
    assert "stage_elapsed_s=" in caplog.text
    assert "'method': 'POST'" in caplog.text
    assert "'path': '/sessions/session-test/v1/messages'" in caplog.text
    assert "private-" not in caplog.text
    if mode == "complete":
        assert sent[-1]["body"] == b"private-response"
        assert "event=response_headers_sending" in caplog.text
        assert "event=response_first_body_sending" in caplog.text


@pytest.mark.parametrize("path,reason", [("messages", "handler_not_found"), ("unknown", "route_not_found")])
def test_not_found_route_diagnostics(path, reason, caplog):
    import httpx
    from fastapi import FastAPI, HTTPException

    caplog.set_level(logging.INFO, logger="gateway.requests")
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.post("/sessions/{session_id}/v1/messages")
    async def missing_session(session_id: str):
        log_request_stage("session_not_found")
        raise HTTPException(status_code=404)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/sessions/test/v1/{path}")
            assert response.status_code == 404

    asyncio.run(run())
    assert f"'not_found_reason': '{reason}'" in caplog.text
    assert ("event=session_not_found" in caplog.text) == (path == "messages")
