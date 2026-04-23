import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG

COLLECTION_NAME = "hs_laws"


class EmbeddingPipeline:
    def __init__(self, azure_model, chroma_collection):
        self.azure_model = azure_model
        self.collection = chroma_collection

        cfg = CONFIG["chunking"]
        self.chunk_size    = cfg["chunk_size"]
        self.chunk_overlap = cfg["chunk_overlap"]
        self._encoder      = tiktoken.get_encoding(cfg["encoding"])

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping token-based chunks.
        Encodes → slides a window of chunk_size with step (chunk_size - chunk_overlap) → decodes.
        """
        tokens = self._encoder.encode(text)
        chunks = []
        step   = self.chunk_size - self.chunk_overlap
        start  = 0

        while start < len(tokens):
            end         = min(start + self.chunk_size, len(tokens))
            chunk_text  = self._encoder.decode(tokens[start:end])
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start += step

        return chunks

    def load_chunks_from_json(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load pages from a preprocessed JSON file and split each into token-sized chunks."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for doc_name, pages in data.items():
            for page_key, page_data in pages.items():
                content = page_data.get("content", "").strip()
                if not content:
                    continue

                base_metadata = {
                    "doc_name":      doc_name,
                    "page":          page_data.get("page", [None])[0],
                    "law_group":     str(page_data.get("law_group") or ""),
                    "version_index": page_data.get("version_index"),
                    "topic":         page_data.get("topic") or "",
                    "year":          page_data.get("year"),
                    "doc_extension": page_data.get("doc_extension", "pdf"),
                }

                sub_chunks = self._split_into_chunks(content)
                for i, sub in enumerate(sub_chunks):
                    chunks.append({
                        "id":       f"{doc_name}__{page_key}__chunk_{i}",
                        "content":  sub,
                        "metadata": {**base_metadata, "chunk_index": i, "chunk_total": len(sub_chunks)},
                    })

        return chunks

    def embed_and_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed a list of chunks and persist them in ChromaDB."""
        texts     = [c["content"]  for c in chunks]
        ids       = [c["id"]       for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        if self.azure_model is not None:
            embeddings = self.azure_model.call_embed_model_batch(texts)
        else:
            embeddings = [[0.0] * 1536 for _ in texts]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  Stored {len(chunks)} chunks ({len(set(c['metadata']['page'] for c in chunks))} pages → {len(chunks)} chunks).")

    def run(self, json_folder: Path) -> None:
        """Embed and store all JSON files produced by the preprocessing stage."""
        json_files = list(json_folder.rglob("*.json"))
        if not json_files:
            print(f"No JSON files found in {json_folder}")
            return

        for json_path in json_files:
            print(f"Processing: {json_path.name}")
            chunks = self.load_chunks_from_json(json_path)
            if chunks:
                self.embed_and_store(chunks)


def build_chroma_collection(db_path: Path, reset: bool = False):
    client = chromadb.PersistentClient(path=str(db_path))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Existing collection '{COLLECTION_NAME}' deleted.")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection
