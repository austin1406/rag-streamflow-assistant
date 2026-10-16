"""Tool-calling agent on top of the retrieval pipeline.

Provider-agnostic on purpose: the model is asked to reply with one JSON object per turn
(either a tool call or a final answer), so it works with every backend in llm_backend.py,
which only expose generate(prompt). The loop validates every reply, runs the requested tool,
feeds the result back, and returns structured output.

Guardrails: tools are whitelisted, arguments are validated, the SQL tool is read-only, the
loop is capped at MAX_STEPS, and any citation that does not match a passage the agent actually
retrieved is dropped and flagged instead of being shown to the user.
"""
import json

from . import tools
from .llm_backend import BackendError, get_backend

MAX_STEPS = 4
MAX_TOOL_RESULT_CHARS = 6000
TOOL_NAMES = ("search_documents", "query_eval_results")

SYSTEM_PROMPT = """You are a research assistant for a hydrology report on LSTM streamflow \
prediction with Dual Adaptive Conformal Prediction (DACP) uncertainty quantification. You can \
call tools before answering.

Reply with exactly ONE JSON object per turn and nothing else.

To call a tool:
{"tool": "<name>", "args": {...}}

To give your final answer:
{"final": {"answer": "<text with inline [n] citations>", "citations": [<n>, ...]}}

Tools:
1. search_documents: args {"query": string, "top_k": integer 1-10}. Returns numbered passages \
from the report, its slide decks, and the training script.
2. query_eval_results: args {"sql": string}. Runs ONE read-only SELECT over the evaluation \
database, which has these tables:
   eval_questions(id, question, expected_source, expected_terms)
   eval_retrieval(question_id, top_k, retrieval_ok, top_sources)

Rules:
- For any question about the research, call search_documents and use ONLY the passages it \
returns. Never use outside knowledge.
- Cite passages by their number, like [1], and list every number you cited in "citations".
- Use query_eval_results only for questions about the evaluation itself (counts, which \
questions passed or failed).
- If the tools do not return the answer, say so in the final answer instead of guessing."""


class AgentFormatError(ValueError):
    pass


def parse_message(raw: str) -> dict:
    start = raw.find("{")
    if start == -1:
        raise AgentFormatError("no JSON object found")
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as e:
        raise AgentFormatError(f"invalid JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise AgentFormatError("reply must be a JSON object")

    if "final" in obj:
        final = obj["final"]
        if not isinstance(final, dict):
            raise AgentFormatError('"final" must be an object')
        answer = final.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise AgentFormatError('"final.answer" must be a non-empty string')
        citations = final.get("citations", [])
        if not isinstance(citations, list) or not all(
            isinstance(c, int) and not isinstance(c, bool) for c in citations
        ):
            raise AgentFormatError('"final.citations" must be a list of integers')
        return {"type": "final", "answer": answer.strip(), "citations": citations}

    if "tool" in obj:
        name = obj["tool"]
        if name not in TOOL_NAMES:
            raise AgentFormatError(f"unknown tool {name!r}; choose from {list(TOOL_NAMES)}")
        args = obj.get("args", {})
        if not isinstance(args, dict):
            raise AgentFormatError('"args" must be an object')
        return {"type": "tool", "name": name, "args": args}

    raise AgentFormatError('JSON must contain either "tool" or "final"')


def _build_prompt(question: str, transcript: list) -> str:
    history = "\n\n".join(transcript)
    return (
        f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\n"
        f"{history}{chr(10) * 2 if history else ''}Your next JSON object:"
    )


def _run_tool(msg: dict, passages: dict, seen: dict, retrieve_fn, db_path):
    """Returns (ok, text_for_model, short_detail)."""
    name, args = msg["name"], msg["args"]
    try:
        if name == "search_documents":
            res = tools.search_documents(args, retrieve_fn=retrieve_fn)
            if not res["ok"]:
                return False, res["error"], res["error"]
            lines = []
            for c in res["chunks"]:
                n = seen.setdefault(c["id"], len(seen) + 1)
                passages[n] = {"n": n, **c}
                lines.append(f"[{n}] (source: {c['source']}, {c['loc']})\n{c['text']}")
            return True, "\n\n".join(lines) or "no passages found", f"{len(res['chunks'])} passages"

        res = tools.query_eval_results(args, db_path=db_path)
        if not res["ok"]:
            return False, res["error"], res["error"]
        text = json.dumps({"columns": res["columns"], "rows": res["rows"], "truncated": res["truncated"]})
        return True, text, f"{len(res['rows'])} rows"
    except Exception as e:
        return False, f"tool failed: {e}", f"tool failed: {e}"


def _finalize(result: dict, msg: dict, passages: dict) -> None:
    valid = []
    for n in dict.fromkeys(msg["citations"]):
        if n in passages:
            valid.append(n)
        else:
            result["warnings"].append(f"dropped citation [{n}]: it does not match any retrieved passage")
    if passages and not valid:
        result["warnings"].append("the answer cites no retrieved passage")

    result["answer"] = msg["answer"]
    result["citations"] = [
        {"n": n, "source": passages[n]["source"], "loc": passages[n]["loc"]} for n in valid
    ]
    result["chunks"] = [passages[n] for n in sorted(passages)]


def run_agent(question, backend=None, retrieve_fn=None, db_path=None, max_steps=MAX_STEPS) -> dict:
    result = {
        "question": question,
        "answer": None,
        "citations": [],
        "chunks": [],
        "tool_calls": [],
        "steps": 0,
        "warnings": [],
        "error": None,
    }
    try:
        backend = backend or get_backend()
    except BackendError as e:
        result["error"] = str(e)
        return result

    passages, seen, transcript = {}, {}, []

    for step in range(1, max_steps + 1):
        result["steps"] = step
        try:
            raw = backend.generate(_build_prompt(question, transcript))
        except BackendError as e:
            result["error"] = str(e)
            return result

        try:
            msg = parse_message(raw)
        except AgentFormatError as e:
            result["warnings"].append(f"step {step}: invalid reply ({e})")
            transcript.append(
                f"Assistant: {raw.strip()[:500]}\nSystem: invalid reply ({e}). "
                "Reply with ONE valid JSON object."
            )
            continue

        if msg["type"] == "final":
            _finalize(result, msg, passages)
            return result

        ok, text, detail = _run_tool(msg, passages, seen, retrieve_fn, db_path)
        result["tool_calls"].append({"tool": msg["name"], "args": msg["args"], "ok": ok, "detail": detail})
        transcript.append(
            f"Assistant: {json.dumps({'tool': msg['name'], 'args': msg['args']})}\n"
            f"Tool result ({'ok' if ok else 'error'}):\n{text[:MAX_TOOL_RESULT_CHARS]}"
        )

    result["error"] = f"the agent did not give a final answer within {max_steps} steps"
    return result
