"""LLM-01: real SDK completion admission and QA publication, no network."""

import contextlib
from contextlib import contextmanager
import io
import json
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import httpx
from openai import OpenAI
from fastapi.testclient import TestClient
from llm.openai_llm_provider import OpenAILLMProvider
from test_context_capacity import make_case
from test_question_output_boundary import route_app

MODELS = ("gpt-4.1-mini", "gpt-4o-mini")
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
PARTIAL = "SYNTHETIC_PARTIAL_PRIVATE_SOURCE"
rows = []


def payload(model, state, text):
    if model == "gpt-4.1-mini":
        content = (
            [{"type": "refusal", "refusal": "Synthetic refusal"}]
            if state == "refusal"
            else [{"type": "output_text", "text": text, "annotations": []}]
        )
        return dict(
            id="resp_audit",
            object="response",
            created_at=0,
            model=model,
            status="incomplete" if state == "incomplete" else "completed",
            incomplete_details=(
                {"reason": "max_output_tokens"} if state == "incomplete" else None
            ),
            output=[
                dict(
                    type="message",
                    id="msg_audit",
                    role="assistant",
                    status="incomplete" if state == "incomplete" else "completed",
                    content=content,
                )
            ],
            usage=dict(
                input_tokens=100,
                input_tokens_details={"cached_tokens": 0},
                output_tokens=5,
                output_tokens_details={"reasoning_tokens": 0},
                total_tokens=105,
            ),
        )
    return dict(
        id="chat_audit",
        object="chat.completion",
        created=0,
        model=model,
        choices=[
            dict(
                index=0,
                finish_reason="length" if state == "incomplete" else "stop",
                message=dict(
                    role="assistant",
                    content=None if state == "refusal" else text,
                    refusal="Synthetic refusal" if state == "refusal" else None,
                ),
            )
        ],
        usage=dict(prompt_tokens=100, completion_tokens=5, total_tokens=105),
    )

    return dict(
        id="chat_audit",
        object="chat.completion",
        created=0,
        model=model,
        choices=[
            dict(
                index=0,
                finish_reason="length" if state == "incomplete" else "stop",
                message=dict(
                    role="assistant",
                    content=None if state == "refusal" else text,
                    refusal="Synthetic refusal" if state == "refusal" else None,
                ),
            )
        ],
        usage=dict(prompt_tokens=100, completion_tokens=5, total_tokens=105),
    )


@contextmanager
def case(model, state, text):
    calls = []
    locals = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(
            {
                "path": request.url.path,
                "body": body,
                "timeout": request.extensions.get("timeout"),
            }
        )
        return httpx.Response(200, json=payload(model, state, text))

    with contextlib.redirect_stdout(io.StringIO()):
        p = OpenAILLMProvider(
            SimpleNamespace(get=lambda: "synthetic-invalid-key"), model=model
        )
    old = getattr(p.llm, "_client", None)
    if old is not None:
        old.close()
    shared = OpenAI(
        api_key="synthetic-invalid-key",
        base_url="https://offline.invalid/v1",
        timeout=p.llm.timeout,
        max_retries=p.llm.max_retries,
        default_headers=p.llm.default_headers,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    p.llm._client = shared
    p.llm.api_base = "https://offline.invalid/v1"

    def factory(**kwargs):
        client = OpenAI(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))
        )
        locals.append(client)
        return client

    try:
        with patch("llm.openai_llm_provider.OpenAIClient", side_effect=factory):
            yield p, calls, locals
    finally:
        shared.close()


def invoke(p, operation):
    if operation == "ordinary":
        return p.complete_text("Synthetic complete answer request")
    if operation == "schema":
        return p.complete_text_with_output_budget(
            "Synthetic structured request",
            p.effective_output_budget + 1,
            response_schema=SCHEMA,
        )
    return p.complete_text_once_with_output_budget(
        "Synthetic once request", p.effective_output_budget + 1
    )


def note(model, state, operation, output, calls, locals, error=None):
    raw = payload(model, state, PARTIAL)
    rows.append(
        dict(
            model=model,
            state=state,
            operation=operation,
            provider_evidence={
                "status": raw.get("status"),
                "incomplete_details": raw.get("incomplete_details"),
                "finish_reason": raw.get("choices", [{}])[0].get("finish_reason"),
            },
            returned=output,
            error=type(error).__name__ if error else None,
            requests=len(calls),
            local_clients_closed=[c.is_closed() for c in locals],
        )
    )


class Boundary(unittest.TestCase):
    def setUp(self):
        guard = patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        )
        guard.start()
        self.addCleanup(guard.stop)

    def test_completed_output_and_schema_controls(self):
        for model in MODELS:
            for operation in ("ordinary", "schema", "once"):
                with self.subTest(model=model, operation=operation), case(
                    model, "completed", ' {"answer":"complete"} '
                ) as (p, calls, locals), contextlib.redirect_stdout(io.StringIO()):
                    original = (p.effective_output_budget, p.llm.max_retries)
                    output = invoke(p, operation)
                    self.assertEqual(output, '{"answer":"complete"}')
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(
                        (p.effective_output_budget, p.llm.max_retries), original
                    )
                    budget = p.effective_output_budget + (operation != "ordinary")
                    body = calls[0]["body"]
                    self.assertEqual(
                        body[
                            (
                                "max_output_tokens"
                                if model == MODELS[0]
                                else "max_completion_tokens"
                            )
                        ],
                        budget,
                    )
                    if operation == "schema":
                        fmt = (
                            body["text"]["format"]
                            if model == MODELS[0]
                            else body["response_format"]["json_schema"]
                        )
                        self.assertEqual(fmt["schema"], SCHEMA)
                        self.assertTrue(fmt["strict"])
                        self.assertEqual(calls[0]["timeout"]["read"], 300.0)
                    if operation == "once":
                        self.assertEqual(len(locals), 1)
                        self.assertTrue(locals[0].is_closed())
                        self.assertEqual(locals[0].max_retries, 0)
                    note(model, "completed", operation, output, calls, locals)

    def test_incomplete_text_is_not_returned_as_success(self):
        for model in MODELS:
            for operation in ("ordinary", "schema", "once"):
                with self.subTest(model=model, operation=operation), case(
                    model, "incomplete", PARTIAL
                ) as (p, calls, locals), contextlib.redirect_stdout(io.StringIO()):
                    error = None
                    output = None
                    try:
                        output = invoke(p, operation)
                    except Exception as exc:
                        error = exc
                    note(model, "incomplete", operation, output, calls, locals, error)
                    self.assertEqual(len(calls), 1)
                    if operation == "once":
                        self.assertEqual(len(locals), 1)
                        self.assertTrue(locals[0].is_closed())
                    self.assertIsNotNone(
                        error,
                        "Explicit provider truncation must not become plain-text success",
                    )

    def test_actual_qa_does_not_publish_incomplete_answer_or_session_progress(self):
        for model in MODELS:
            with self.subTest(model=model), case(model, "incomplete", PARTIAL) as (
                p,
                calls,
                locals,
            ), contextlib.redirect_stdout(io.StringIO()):
                c = make_case()
                c.coordinator.llm_provider = p
                with TestClient(route_app(c.coordinator)) as client:
                    r = client.post(
                        "/documents/ask",
                        json={
                            "doc_name": "synthetic",
                            "query": c.query,
                            "session_id": "one",
                        },
                    )
                state = c.coordinator.get_session("one")
                rows.append(
                    dict(
                        model=model,
                        operation="actual_QA_HTTP",
                        state="incomplete",
                        http_status=r.status_code,
                        http_json=r.json(),
                        session_questions=state.recent_questions,
                        session_anchor=state.active_chunk_index,
                        requests=len(calls),
                    )
                )
                self.assertEqual(len(calls), 1)
                self.assertNotIn(PARTIAL, r.text)
                with self.subTest(field="http_success"):
                    self.assertNotEqual(
                        r.status_code,
                        200,
                        "Partial provider output must not be a normal successful answer",
                    )
                with self.subTest(field="session_history"):
                    self.assertEqual(state.recent_questions, [])

    def test_actual_qa_completed_answer_positive_controls(self):
        for model in MODELS:
            with self.subTest(model=model), case(
                model, "completed", "Complete controlled answer"
            ) as (p, calls, locals), contextlib.redirect_stdout(io.StringIO()):
                c = make_case()
                c.coordinator.llm_provider = p
                with TestClient(route_app(c.coordinator)) as client:
                    r = client.post(
                        "/documents/ask",
                        json={
                            "doc_name": "synthetic",
                            "query": c.query,
                            "session_id": "one",
                        },
                    )
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["answer"], "Complete controlled answer")
                self.assertEqual(
                    c.coordinator.get_session("one").recent_questions, [c.query]
                )
                self.assertEqual(len(calls), 1)

    def test_refusal_observation_qualified_separately(self):
        for model in MODELS:
            for operation in ("ordinary", "schema", "once"):
                with case(model, "refusal", "") as (
                    p,
                    calls,
                    locals,
                ), contextlib.redirect_stdout(io.StringIO()):
                    output = None
                    error = None
                    try:
                        output = invoke(p, operation)
                    except Exception as exc:
                        error = exc
                    note(model, "refusal", operation, output, calls, locals, error)
                    self.assertEqual(len(calls), 1)
                    if operation == "once":
                        self.assertTrue(locals[0].is_closed())


class CompletionMatrix(unittest.TestCase):
    def setUp(self):
        guard = patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        )
        guard.start()
        self.addCleanup(guard.stop)

    def check_rejected(self, model, envelope):
        for operation in ("ordinary", "schema", "once"):
            with self.subTest(operation=operation), patch(
                __name__ + ".payload", return_value=envelope
            ), case(model, "completed", PARTIAL) as (
                p,
                calls,
                clients,
            ), contextlib.redirect_stdout(
                io.StringIO()
            ) as logs:
                with self.assertRaises(ValueError) as caught:
                    invoke(p, operation)
                self.assertIn(
                    str(caught.exception),
                    {
                        "completion_incomplete",
                        "completion_status_unavailable",
                        "completion_refused",
                        "completion_text_unavailable",
                    },
                )
                self.assertNotIn(PARTIAL, str(caught.exception) + logs.getvalue())
                self.assertEqual(len(calls), 1)
                if operation == "once":
                    self.assertEqual(len(clients), 1)
                    self.assertTrue(clients[0].is_closed())

    def test_responses_unfinished_status_matrix(self):
        for state in (
            "incomplete",
            "failed",
            "cancelled",
            "queued",
            "in_progress",
            None,
            "unrecognized",
        ):
            with self.subTest(state=state):
                body = payload(MODELS[0], "completed", PARTIAL)
                body["status"] = state
                self.check_rejected(MODELS[0], body)

    def test_responses_incomplete_details_and_message_status(self):
        for reason in ("max_output_tokens", "content_filter"):
            with self.subTest(reason=reason):
                body = payload(MODELS[0], "completed", PARTIAL)
                body["incomplete_details"] = {"reason": reason}
                self.check_rejected(MODELS[0], body)
        body = payload(MODELS[0], "completed", PARTIAL)
        body["output"][0]["status"] = "incomplete"
        self.check_rejected(MODELS[0], body)
        body = payload(MODELS[0], "completed", PARTIAL)
        body["error"] = {"code": "server_error", "message": PARTIAL}
        self.check_rejected(MODELS[0], body)

    def test_chat_unfinished_reason_matrix(self):
        for reason in (
            "length",
            "content_filter",
            "tool_calls",
            "function_call",
            None,
            "unrecognized",
        ):
            with self.subTest(reason=reason):
                body = payload(MODELS[1], "completed", PARTIAL)
                body["choices"][0]["finish_reason"] = reason
                self.check_rejected(MODELS[1], body)

    def test_refusal_and_empty_are_never_success(self):
        for model in MODELS:
            for state, text in [
                ("refusal", ""),
                ("completed", ""),
                ("completed", " \n\t"),
            ]:
                with self.subTest(model=model, state=state, text=text):
                    self.check_rejected(model, payload(model, state, text))

    def test_parseable_schema_does_not_override_incomplete_status(self):
        for model in MODELS:
            body = payload(
                model, "incomplete", '{"answer":"parseable but interrupted"}'
            )
            with self.subTest(model=model):
                self.check_rejected(model, body)

    def test_refusal_with_text_is_rejected(self):
        for model in MODELS:
            body = payload(model, "completed", PARTIAL)
            if model == MODELS[0]:
                body["output"][0]["content"].append(
                    {"type": "refusal", "refusal": PARTIAL}
                )
            else:
                body["choices"][0]["message"]["refusal"] = PARTIAL
            with self.subTest(model=model):
                self.check_rejected(model, body)


if __name__ == "__main__":
    unittest.main()
