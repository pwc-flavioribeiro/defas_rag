import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG

COLLECTION_NAME = "hs_laws"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_COMPARISON_SYSTEM = (
    "You are a precise Health & Safety legal analyst. "
    "Follow the output format exactly. Never invent provisions not present in the excerpts."
)

_COMPARISON_USER = """\
TASK
----
Compare the OLD VERSION and NEW VERSION excerpts below and produce a structured legal analysis.

RULES
-----
1. Compare by equivalent legal units ONLY: Article, Annex, Part, Point / sub-point.
2. Do NOT flag purely editorial differences (punctuation, numbering format, layout) unless
   they affect legal meaning.
3. Do NOT produce false positives from excerpt misalignment. If a provision appears in one
   version but cannot be found in the other, label it Added or Removed — never assume it exists.
4. "Modified" is NOT an allowed Change Type. Split every modification into:
     • One row  →  Change Type = Removed  (old wording / concept)
     • One row  →  Change Type = Added    (new wording / concept)
5. Use conservative reasoning: only flag a change when you have textual evidence from BOTH versions.
6. If uncertain, state the uncertainty explicitly in the Explanation column.

ALLOWED VALUES
--------------
Change Type      : Added | Removed | Unchanged
Practical Impact : None | Low | Medium | High
Materiality      : Editorial | Procedural | Substantive

DECISION RULES
--------------
Unchanged   → same legal meaning in both versions
Added       → genuinely new provision or wording not present before
Removed     → provision or wording present before and absent now
None        → no operational impact
Low         → limited or indirect operational impact
Medium      → requires process / documentation / control updates
High        → affects legal obligations, thresholds, exposure limits, or immediate compliance duties
Editorial   → affects only form, not legal substance
Procedural  → affects administrative or process obligations
Substantive → affects rights, duties, thresholds, or legal obligations

OUTPUT FORMAT
-------------
1. Start with a concise executive summary (3 - 5 sentences) covering the overall scope of changes.
2. Then produce a markdown table with EXACTLY these columns:

| Legal Unit | Topic | Old Version | New Version | Change Type | Practical Impact | Materiality | Explanation |
|---|---|---|---|---|---|---|---|

Rules for table cells:
- Legal Unit  : the specific Article / Annex / Part / Point being compared (e.g. "Article 3", "Annex I")
- Topic       : short label for the provision subject (e.g. "Exposure limit values", "Definitions")
- Old Version : verbatim or summarised wording from the {old_year} version; "—" if not present
- New Version : verbatim or summarised wording from the {new_year} version; "—" if not present
- Explanation : cite textual evidence from both versions for each conclusion
{intermediate_note}
QUESTION
--------
{question}

OLD VERSION ({old_year})
------------------------
{old_context}

NEW VERSION ({new_year})
------------------------
{new_context}
"""

_STANDARD_SYSTEM = (
    "You are a precise EU Health & Safety legal analyst. "
    "Answer using only the provided context. If context is insufficient, say so explicitly."
)

_STANDARD_USER = """\
QUESTION
--------
{question}

CONTEXT
-------
{context}
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class QueryPipeline:
    def __init__(self, azure_model, chroma_collection):
        self.azure_model   = azure_model
        self.collection    = chroma_collection
        self.n_single      = CONFIG["retrieval"]["n_results"]
        self.n_comparison  = CONFIG["retrieval"]["n_results_comparison"]

    # ── Embedding helper ────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        if self.azure_model is not None:
            return self.azure_model.call_embed_model(text)
        return [0.0] * 1536   # placeholder when credentials are absent

    # ── ChromaDB helpers ────────────────────────────────────────────────────

    def _get_available_versions(self, law_group: str) -> List[int]:
        """Return sorted list of years stored for a given law group."""
        result = self.collection.get(
            where={"law_group": {"$eq": law_group}},
            include=["metadatas"],
        )
        years = sorted({m["year"] for m in result["metadatas"] if m.get("year")})
        return years

    def _query_collection(self, vector: List[float], n: int, where: Optional[Dict]) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"content": doc, "metadata": meta, "score": 1 - dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def retrieve(self, question: str, law_group: Optional[str] = None) -> List[Dict]:
        """General retrieval for single-version or unfiltered queries."""
        vector = self._embed(question)
        where  = {"law_group": {"$eq": law_group}} if law_group else None
        return self._query_collection(vector, self.n_single, where)

    def _retrieve_per_version(self, question: str, law_group: str, year: int) -> List[Dict]:
        """Retrieve chunks for one specific law group + year, sorted by page and chunk order."""
        vector = self._embed(question)
        where  = {"$and": [{"law_group": {"$eq": law_group}}, {"year": {"$eq": year}}]}
        chunks = self._query_collection(vector, self.n_comparison, where)
        return sorted(chunks, key=lambda c: (
            c["metadata"].get("page", 0),
            c["metadata"].get("chunk_index", 0),
        ))

    # ── Context formatting ──────────────────────────────────────────────────

    @staticmethod
    def _format_context(chunks: List[Dict]) -> str:
        return "\n\n---\n\n".join(c["content"] for c in chunks)

    # ── Prompt builders ─────────────────────────────────────────────────────

    def _comparison_messages(
        self,
        question: str,
        chunks_by_year: Dict[int, List[Dict]],
        all_years: List[int],
    ) -> List[Dict]:
        old_year = all_years[0]
        new_year = all_years[-1]

        intermediate = [y for y in all_years if y not in (old_year, new_year)]
        intermediate_note = (
            f"Note: intermediate version(s) {intermediate} exist between {old_year} and {new_year}. "
            "Flag in Explanation if a change may have originated in an intermediate version.\n"
            if intermediate else ""
        )

        user = _COMPARISON_USER.format(
            question=question,
            old_year=old_year,
            new_year=new_year,
            old_context=self._format_context(chunks_by_year[old_year]),
            new_context=self._format_context(chunks_by_year[new_year]),
            intermediate_note=intermediate_note,
        )
        return [
            {"role": "system", "content": _COMPARISON_SYSTEM},
            {"role": "user",   "content": user},
        ]

    def _standard_messages(self, question: str, chunks: List[Dict]) -> List[Dict]:
        user = _STANDARD_USER.format(
            question=question,
            context=self._format_context(chunks),
        )
        return [
            {"role": "system", "content": _STANDARD_SYSTEM},
            {"role": "user",   "content": user},
        ]

    # ── Main entry points ────────────────────────────────────────────────────

    def run(self, question: str, law_group: Optional[str] = None) -> str:
        answer, _ = self.run_with_sources(question, law_group=law_group)
        return answer

    def run_with_sources(
        self, question: str, law_group: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Full RAG pipeline.
        - If a law_group with multiple versions is detected → comparison mode.
        - Otherwise → standard RAG.
        Returns (answer, all_chunks_used).
        """
        # Resolve which law group to use
        target_group = law_group
        if not target_group:
            # Peek at top results to infer law group
            probe = self.retrieve(question)
            groups = {c["metadata"].get("law_group") for c in probe if c["metadata"].get("law_group")}
            if len(groups) == 1:
                target_group = list(groups)[0]

        if target_group:
            years = self._get_available_versions(target_group)
        else:
            years = []

        # ── Comparison mode ─────────────────────────────────────────────────
        if len(years) > 1:
            chunks_by_year = {
                year: self._retrieve_per_version(question, target_group, year)
                for year in years
            }
            messages   = self._comparison_messages(question, chunks_by_year, years)
            all_chunks = [c for v in chunks_by_year.values() for c in v]

        # ── Standard mode ────────────────────────────────────────────────────
        else:
            all_chunks = (
                self._retrieve_per_version(question, target_group, years[0])
                if years
                else self.retrieve(question, law_group=law_group)
            )
            messages = self._standard_messages(question, all_chunks)

        if not all_chunks:
            return "No relevant content found in the knowledge base.", []

        if self.azure_model is not None:
            answer = self.azure_model.call_generation_model(messages, max_token=16000)
        else:
            preview = "\n".join(f"[{m['role']}]: {m['content'][:300]}..." for m in messages)
            answer  = (
                "[Credentials not configured — LLM call skipped]\n\n"
                f"Prompt that would be sent:\n{preview}"
            )

        return answer, all_chunks


# ---------------------------------------------------------------------------
# ChromaDB factory
# ---------------------------------------------------------------------------

def build_chroma_collection(db_path: Path, reset: bool = False):
    client = chromadb.PersistentClient(path=str(db_path))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
