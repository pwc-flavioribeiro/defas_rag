"""
Query pipeline — ask a question against the embedded H&S law documents.

Usage:
    python run_query.py "What are the noise exposure limits?"
    python run_query.py "What changed in the vibration directive?" --law-group 1
"""
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "3.Query"))

from query import QueryPipeline, build_chroma_collection
from config import CONFIG


def get_azure_model():
    """Return a configured Azure model client, or None if credentials are missing."""
    try:
        from utils.azure_openai_models_utils import AsyncAzureOpenAIModels
        return AsyncAzureOpenAIModels().initialize()
    except ValueError as e:
        print(f"[WARNING] Azure credentials not configured — LLM calls will be skipped.\n  {e}\n")
        return None


def run_query(question: str, law_group: str = None):
    azure_model = get_azure_model()
    collection  = build_chroma_collection(CONFIG["folder"]["chroma_db"])
    pipeline    = QueryPipeline(azure_model=azure_model, chroma_collection=collection)

    answer = pipeline.run(question, law_group=law_group)

    print(f"\nQuestion : {question}")
    if law_group:
        print(f"Law group: {law_group}")
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the H&S law RAG system.")
    parser.add_argument("question", help="Natural language question to ask.")
    parser.add_argument(
        "--law-group",
        default=None,
        help="Restrict retrieval to a specific law group (e.g. '1' for vibration, '2' for noise).",
    )
    args = parser.parse_args()
    run_query(args.question, law_group=args.law_group)
