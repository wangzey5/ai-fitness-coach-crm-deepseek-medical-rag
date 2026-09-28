from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter
from typing import Any

from .database import Database


VECTOR_SIZE = 384
COLLECTION_ALIASES = {
    "needs": "market",
    "vehicles": "course",
    "workouts": "course",
    "medical": "market",
    "market": "market",
    "course": "course",
}


def _tokens(text: str) -> list[str]:
    normalized = text.lower().strip()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    chinese = "".join(token for token in words if len(token) == 1 and "\u4e00" <= token <= "\u9fff")
    bigrams = [chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))]
    return words + bigrams


def local_embedding(text: str) -> list[float]:
    counts: Counter[int] = Counter()
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % VECTOR_SIZE
        counts[index] += 1
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return [counts.get(index, 0) / norm for index in range(VECTOR_SIZE)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class VectorStore:
    def __init__(self, database: Database):
        self.database = database

    def upsert(
        self,
        collection: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> str:
        normalized_collection = COLLECTION_ALIASES.get(collection)
        if not normalized_collection:
            raise ValueError("collection must be 'needs' or 'workouts'")
        document_id = document_id or str(uuid.uuid4())
        metadata = {**(metadata or {}), "_logical_collection": collection}
        self.database.upsert_document(
            document_id,
            normalized_collection,
            title,
            content,
            metadata,
            local_embedding(f"{title}\n{content}"),
        )
        return document_id

    def search(self, collection: str, query: str, limit: int = 3) -> list[dict[str, Any]]:
        normalized_collection = COLLECTION_ALIASES.get(collection)
        if not normalized_collection:
            raise ValueError("collection must be 'needs' or 'workouts'")
        query_vector = local_embedding(query)
        documents = self.database.list_documents(normalized_collection)
        filtered = [
            document
            for document in documents
            if document["metadata"].get("_logical_collection", collection) == collection
        ]
        scored = [
            {**document, "score": round(cosine_similarity(query_vector, document["vector"]), 4)}
            for document in filtered
        ]
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]
