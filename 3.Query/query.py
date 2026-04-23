import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG

COLLECTION_NAME = "hs_laws"


class QueryPipeline:
    def __init__(self, azure_model, chroma_collection):
        self.azure_model = azure_model
        self.collection = chroma_collection

    def retrieve(self, question: str, law_group: Optional[str] = None) -> List[Dict[str, Any]]:
        """Embed the question and retrieve the top-N most relevant chunks."""
        n_results = CONFIG["retrieval"]["n_results"]

        if self.azure_model is not None:
            # Real path — requires credentials in .env
            query_vector = self.azure_model.call_embed_model(question)
        else:
            # Placeholder: zero vector so the pipeline can be tested without credentials
            query_vector = [0.0] * 1536

        where = {"law_group": law_group} if law_group else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({"content": doc, "metadata": meta, "score": 1 - dist})
        return chunks

    def _has_multiple_versions(self, chunks: List[Dict]) -> bool:
        years = {c["metadata"].get("year") for c in chunks if c["metadata"].get("year")}
        return len(years) > 1

    def _build_prompt(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Build the LLM messages list, using a comparison prompt when multiple law versions are found."""
        if self._has_multiple_versions(chunks):
            by_year: Dict[int, List[str]] = {}
            for c in chunks:
                year = c["metadata"].get("year", "unknown")
                by_year.setdefault(year, []).append(c["content"])

            context_parts = []
            for year in sorted(by_year.keys()):
                joined = "\n\n".join(by_year[year])
                context_parts.append(f"=== Version {year} ===\n{joined}")
            context = "\n\n".join(context_parts)

            system = (
                "You are an expert in Health & Safety law. "
                "You are given excerpts from multiple versions of a law directive. "
                "Compare the versions and clearly highlight what changed, what was added, and what was removed."
            )
            user = (
                f"Question: {question}\n\n"
                f"Law excerpts by version:\n{context}\n\n"
                "Provide a structured comparison of the changes between versions."
            )
        else:
            context = "\n\n".join(c["content"] for c in chunks)
            system = (
                "You are an expert in Health & Safety law. "
                "Answer the question using only the provided context. "
                "If the context does not contain enough information, say so."
            )
            user = f"Question: {question}\n\nContext:\n{context}"

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def run(self, question: str, law_group: Optional[str] = None) -> str:
        """Full RAG pipeline: retrieve → build prompt → generate answer."""
        answer, _ = self.run_with_sources(question, law_group=law_group)
        return answer

    def run_with_sources(self, question: str, law_group: Optional[str] = None):
        """Full RAG pipeline returning (answer, chunks) so callers can display sources."""
        chunks = self.retrieve(question, law_group=law_group)

        if not chunks:
            return "No relevant content found in the knowledge base.", []

        messages = self._build_prompt(question, chunks)

        if self.azure_model is not None:
            answer = self.azure_model.call_generation_model(messages)
        else:
            prompt_preview = "\n".join(f"[{m['role']}]: {m['content'][:200]}..." for m in messages)
            answer = (
                "[Credentials not configured — LLM call skipped]\n\n"
                f"Prompt that would be sent:\n{prompt_preview}"
            )

        return answer, chunks


def build_chroma_collection(db_path: Path):
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection
