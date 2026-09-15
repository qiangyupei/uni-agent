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
                {"type": "http", "path": "/sessions/session-test/v1/messages"}, receive, send
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
    assert "private-" not in caplog.text
    if mode == "complete":
        assert sent[-1]["body"] == b"private-response"
        assert "event=response_headers_sending" in caplog.text
        assert "event=response_first_body_sending" in caplog.text
