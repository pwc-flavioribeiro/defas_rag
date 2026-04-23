import os
import re
import json
import httpx
from openai import AzureOpenAI
from typing import TYPE_CHECKING, Dict, Any, Optional, List


FUNCTION = "OPENAI MODEL:"

class AsyncAzureOpenAIModels:
    """
    Wrapper around Azure OpenAI configuration and usage.

    Responsibilities:
      - Load credentials from environment variables.
      - Build a dynamic configuration per `model_name` (api_version, deployment).
      - Create Azure OpenAI clients on demand.
      - Call the chat completion API with parameters tailored per model tier.
    """

    SUPPORTED_MODELS = ("HIGH_MODEL", "MEDIUM_MODEL", "LOW_MODEL")

    def __init__(self):
        """
        Initialize the wrapper with a logger and load environment-based settings.

        Args:
            log_system: LoggerSystem instance used for structured logging.

        Raises:
            ValueError: If mandatory environment variables are missing.
        """
        self.azure_client: Optional[AzureOpenAI] = None
        # Environment placeholders
        self.AZURE_OPENAI_ENDPOINT: Optional[str] = None
        self.AZURE_OPENAI_API_KEY: Optional[str] = None
        self.AZURE_OPENAI_EMBEDDING_MODEL: Optional[str] = None
        self.AZURE_OPENAI_API_VERSION: Optional[str] = None

        # Per-model configuration map
        self._model_config: Dict[str, Dict[str, str]] = {}
    
    def initialize(self):
        self.load_azureopenai_credential()
        self._build_model_config()
        return self

    def load_azureopenai_credential(self) -> None:
        """
        Load all required environment variables and perform minimal validation.

        Required variables:
          - AZURE_OPENAI_ENDPOINT
          - AZURE_OPENAI_KEY
          - AZURE_OPENAI_API_VERSION (global fallback)
          - AZURE_OPENAI_EMBEDDING_MODEL
          - AZURE_OPENAI_MODEL_HIGH_MODEL
          - AZURE_OPENAI_MODEL_MEDIUM_MODEL
          - AZURE_OPENAI_MODEL_LOW_MODEL

        Optional (per-tier API versions; falls back to the global version if missing):
          - AZURE_OPENAI_API_VERSION_HIGH_MODEL
          - AZURE_OPENAI_API_VERSION_MEDIUM_MODEL
          - AZURE_OPENAI_API_VERSION_LOW_MODEL

        Raises:
            ValueError: If mandatory variables are not present.
        """
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")     
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
        self.AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        self.AZURE_OPENAI_API_VERSION_EMBEDDING = os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING")
        self.AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

        missing: List[str] = []
        if not self.AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_KEY")
        if not self.AZURE_OPENAI_API_VERSION:
            missing.append("AZURE_OPENAI_API_VERSION")
        if not self.AZURE_OPENAI_EMBEDDING_ENDPOINT:
            missing.append("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        if not self.AZURE_OPENAI_EMBEDDING_MODEL:
            missing.append("AZURE_OPENAI_EMBEDDING_MODEL")
        if not self.AZURE_OPENAI_MODEL:
            missing.append("AZURE_OPENAI_MODEL")


        if missing:
            raise ValueError(
                f"Missing Azure OpenAI environment variables: {', '.join(missing)}"
            )

    def _build_model_config(self) -> None:
        """
        Build the configuration map per model tier (api_version and deployment),
        using per-tier API versions and model when available, otherwise falling back to the global version.
        """
        self._model_config = {
            "LLM_MODEL": {
                "api_version": self.AZURE_OPENAI_API_VERSION,
                "deployment": self.AZURE_OPENAI_MODEL,
                "endpoint":  self.AZURE_OPENAI_ENDPOINT
            },
            "EMBED_MODEL": {
                "api_version": self.AZURE_OPENAI_API_VERSION_EMBEDDING,
                "deployment": self.AZURE_OPENAI_EMBEDDING_MODEL,
                "endpoint": self.AZURE_OPENAI_EMBEDDING_ENDPOINT
            },
        }

    def _get_model_config(self, model_name: str) -> Dict[str, str]:
        """
        Retrieve the configuration for the given model name.

        Args:
            model_name: One of "HIGH_MODEL", "MEDIUM_MODEL", "LOW_MODEL".

        Returns:
            A dict containing:
              - 'api_version': The API version string to use.
              - 'deployment': The Azure OpenAI deployment name to call.

        Raises:
            ValueError: If the requested model name is not supported.
        """
        key = (model_name or "").upper()
        cfg = self._model_config.get(key)
        if not cfg:
            raise ValueError(
                f"Model name '{model_name}' is not supported. "
                f"Use one of: {', '.join(self.SUPPORTED_MODELS)}"
            )
        return cfg

    def create_azure_openai_client(self, model_name: str) -> None:
        """
        Initialize the AzureOpenAI client for the requested model tier.

        The client uses the `api_version` associated with `model_name`,
        falling back to the global version if a per-tier version is missing.

        Args:azure_client
            model_name: Model tier identifier (HIGH_MODEL | MEDIUM_MODEL | LOW_MODEL).

        Raises:
            Exception: If client instantiation fails for any reason.
        """
        try:
            self.cfg = self._get_model_config(model_name)
            # Corporate networks (e.g. Zscaler) use TLS inspection with self-signed certs.
            # Set AZURE_OPENAI_VERIFY_SSL=false in .env to disable verification in those environments.
            verify_ssl = os.getenv("AZURE_OPENAI_VERIFY_SSL", "true").lower() != "false"
            self.azure_client = AzureOpenAI(
                api_key=self.AZURE_OPENAI_API_KEY,
                api_version=self.cfg["api_version"],
                azure_endpoint=self.cfg["endpoint"],
                http_client=httpx.Client(verify=verify_ssl),
            )
        except Exception as e:
            raise e

    def call_generation_model(
            self,
            messages: list,
            max_token: int = 10000,
            model_name: str = "LLM_MODEL",
        ) -> str:
            """
            Execute a chat completion using the specified model tier.

            Behavior:
              - Builds/refreshes the client using the model's configured API version.
              - Uses 'max_completion_tokens' and 'reasoning_effort' for HIGH/MEDIUM tiers.
              - Uses 'temperature' and 'max_tokens' for the LOW tier.

            Args:
                messages: List of chat messages (role/content dicts) to send to the model.
                max_token: Max tokens for the completion (interpreted as max_completion_tokens for HIGH/MEDIUM; max_tokens for LOW).
                call_type: Optional label for logging/trace purposes.
                model_name: Model tier ("HIGH_MODEL" | "MEDIUM_MODEL" | "LOW_MODEL"). Defaults to "HIGH_MODEL".
                reasoning_effort: Reasoning effort level for models that support it (e.g., 'minimal', 'medium', 'high').

            Returns:
                The string content of the first choice message.

            Raises:
                Exception: If the completion request fails.
            """

            self.create_azure_openai_client(model_name= model_name)

            try:
                deployment = self.cfg["deployment"]

                response = self.azure_client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_completion_tokens=max_token,
                )
                    
                return response.choices[0].message.content.strip()

            except Exception as e:
                raise e

    def call_embed_model(self, query: str) -> List[float]:
        self.create_azure_openai_client(model_name="EMBED_MODEL")
        try:
            response = self.azure_client.embeddings.create(
                input=query, model=self.AZURE_OPENAI_EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            raise e

    def call_embed_model_batch(self, texts: List[str]) -> List[List[float]]:
        self.create_azure_openai_client(model_name="EMBED_MODEL")
        try:
            response = self.azure_client.embeddings.create(
                input=texts, model=self.AZURE_OPENAI_EMBEDDING_MODEL
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise e
