"""
Ingestion pipeline — run this once to process all PDFs and populate ChromaDB.

Usage:
    python run_ingestion.py
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Numbered folders are not valid Python module names, so we add them to sys.path
sys.path.insert(0, str(Path(__file__).parent / "1.Preprossessing"))
sys.path.insert(0, str(Path(__file__).parent / "2.Embedding"))

from pdf_preprocessing import PdfProcessor
from embedding_pipeline import EmbeddingPipeline, build_chroma_collection
from config import CONFIG, PROJECT_ROOT

PREPROCESSOR_CONFIG = {
    "folder": {
        "txt_output":  CONFIG["folder"]["txt_output"],
        "json_output": CONFIG["folder"]["json_output"],
    }
}


def get_azure_model():
    """Return a configured Azure model client, or None if credentials are missing."""
    try:
        from utils.azure_openai_models_utils import AsyncAzureOpenAIModels
        return AsyncAzureOpenAIModels().initialize()
    except ValueError as e:
        print(f"[WARNING] Azure credentials not configured — embeddings will use placeholder vectors.\n  {e}\n")
        return None


def run_ingestion():
    azure_model = get_azure_model()
    collection  = build_chroma_collection(CONFIG["folder"]["chroma_db"], reset=True)
    preprocessor = PdfProcessor(config=PREPROCESSOR_CONFIG)
    embedder     = EmbeddingPipeline(azure_model=azure_model, chroma_collection=collection)

    pdf_files = list(CONFIG["folder"]["docs"].glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {CONFIG['folder']['docs']}")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    json_paths = []
    for pdf_path in pdf_files:
        print(f"--- {pdf_path.name} ---")

        txt_path = preprocessor.process_text(
            pdf_path=str(pdf_path),
            project_root=PROJECT_ROOT,
        )
        print(f"  Text extracted  → {txt_path}")

        json_path = preprocessor.from_text_to_json(
            output_text_path=str(txt_path),
            project_root=PROJECT_ROOT,
            pdf_name=pdf_path.name,
        )
        print(f"  JSON created    → {json_path}")
        json_paths.append(json_path)

    print("\n--- Embedding and storing chunks in ChromaDB ---")
    embedder.run(PROJECT_ROOT / CONFIG["folder"]["json_output"])

    print("\nIngestion complete.")


if __name__ == "__main__":
    run_ingestion()
