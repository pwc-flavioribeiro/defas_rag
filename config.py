from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

CONFIG = {
    "folder": {
        "docs":        PROJECT_ROOT / "Docs",
        "txt_output":  "output/txt",
        "json_output": "output/json",
        "chroma_db":   PROJECT_ROOT / "output" / "chroma_db",
    },
    "chunking": {
        "chunk_size":    550,   # tokens  (target: 500-600)
        "chunk_overlap": 100,   # tokens  (target: 80-120)
        "encoding":      "cl100k_base",  # encoding used by text-embedding-ada-002
    },
    "retrieval": {
        "n_results":             5,   # chunks for single-version queries
        "n_results_comparison":  3,   # chunks per version for comparison (kept low — reasoning model needs output budget)
    },
}
