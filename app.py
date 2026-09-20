"""Flask front end for the streamflow report RAG assistant.

Run with:
    python app.py
Then open http://localhost:5000
"""
from flask import Flask, render_template, request, jsonify

from rag.rag import answer
from rag.llm_backend import get_backend, BackendError

app = Flask(__name__)


@app.route("/")
def index():
    try:
        backend = get_backend()
        healthy = backend.health_check() if hasattr(backend, "health_check") else True
        if healthy:
            provider, backend_error = backend.name, None
        else:
            provider, backend_error = None, f"{backend.name} backend is configured but not reachable."
    except BackendError as e:
        provider = None
        backend_error = str(e)
    return render_template("index.html", provider=provider, backend_error=backend_error)


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    result = answer(question, top_k=int(data.get("top_k", 5)))
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
