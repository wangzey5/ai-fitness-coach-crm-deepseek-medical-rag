from __future__ import annotations

import json
import re
from collections.abc import Iterator

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from .config import Settings
from .database import Database
from .llm import LLMService
from .service import ChatService
from .sql_agent import ReadOnlySQLAgent
from .vector_store import VectorStore
from .web_search import WebSearchService


DEFAULT_CUSTOMER_ID = "local-customer"
PHONE_TAIL_PATTERN = re.compile(r"^\d{4}$")


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    vector_store = VectorStore(database)
    search_service = WebSearchService()
    sql_agent = ReadOnlySQLAgent(database)
    llm = LLMService(
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        deepseek_api_key=settings.deepseek_api_key,
        deepseek_model=settings.deepseek_model,
        deepseek_base_url=settings.deepseek_base_url,
    )
    chat_service = ChatService(database, vector_store, sql_agent, llm, search_service)

    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.extensions["crm"] = {
        "database": database,
        "vector_store": vector_store,
        "search_service": search_service,
        "sql_agent": sql_agent,
        "chat_service": chat_service,
    }

    @app.get("/")
    def index():
        return render_template("index.html", openai_enabled=llm.enabled)

    @app.get("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "llm_enabled": llm.enabled, "llm_provider": llm.provider})

    def profile_from_payload(data: dict) -> tuple[str, str | None, str | None]:
        name = str(data.get("name", "")).strip()
        phone_tail = str(data.get("phone_tail", "")).strip()
        if not name and not phone_tail:
            return DEFAULT_CUSTOMER_ID, None, None
        if not name or len(name) > 40 or not PHONE_TAIL_PATTERN.fullmatch(phone_tail):
            raise ValueError("name or phone_tail is invalid")
        return f"profile:{phone_tail}:{name}", name, phone_tail

    @app.post("/api/session/start")
    def start_session():
        data = request.get_json(silent=True) or {}
        try:
            customer_id, name, phone_tail = profile_from_payload(data)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        user, is_new = database.ensure_user(customer_id, name=name, phone_tail=phone_tail)
        return jsonify({"customer_id": customer_id, "is_new_user": is_new, "user": user})

    @app.post("/api/session/end")
    def end_session():
        return jsonify({"status": "ended"})

    @app.post("/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        try:
            customer_id, name, phone_tail = profile_from_payload(data)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        query = str(data.get("query", "")).strip()
        if not query or len(query) > 4000:
            return jsonify({"error": "query is invalid"}), 400
        result = chat_service.interview(customer_id, query, name=name, phone_tail=phone_tail)

        def stream_response() -> Iterator[str]:
            yield json.dumps({"type": "meta", "intent_code": result["intent_code"]}, ensure_ascii=False) + "\n"
            response = result["response"]
            for index in range(0, len(response), 32):
                yield json.dumps({"type": "delta", "text": response[index : index + 32]}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done", "data": result}, ensure_ascii=False) + "\n"

        return Response(stream_with_context(stream_response()), content_type="application/x-ndjson; charset=utf-8")

    @app.post("/api/vectorstores/<collection>/documents")
    def update_vectorstore(collection: str):
        data = request.get_json(silent=True) or {}
        title = str(data.get("title", "")).strip()
        content = str(data.get("content", "")).strip()
        if collection not in {"needs", "workouts", "medical", "vehicles", "market", "course"} or not title or not content:
            return jsonify({"error": "collection, title or content is invalid"}), 400
        document_id = vector_store.upsert(
            collection=collection,
            title=title,
            content=content,
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
            document_id=data.get("id"),
        )
        return jsonify({"id": document_id, "collection": collection}), 201

    @app.post("/api/sql/query")
    def sql_query():
        data = request.get_json(silent=True) or {}
        try:
            rows = sql_agent.execute(str(data.get("sql", "")), data.get("parameters") or [])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"rows": rows})

    return app
