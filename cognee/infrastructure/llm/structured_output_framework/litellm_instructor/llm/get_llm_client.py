"""Get the LLM client."""

import os
from enum import Enum
from typing import Annotated, Any, Literal, Type, Union, get_args, get_origin

from pydantic_core import PydanticUndefined

from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter import (
    OllamaAPIAdapter,
)
from cognee.infrastructure.llm.exceptions import (
    LLMAPIKeyNotSetError,
    UnsupportedLLMProviderError,
)


def _should_use_mock_client() -> bool:
    flag = os.environ.get("LLM_MOCK_RESPONSES", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _coerce_default_value(annotation: Any) -> Any:
    """Derive a reasonable default value for the given annotation."""

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated and args:
        return _coerce_default_value(args[0])

    if origin is Union and args:
        for arg in args:
            if arg is not type(None):  # noqa: E721
                return _coerce_default_value(arg)
        return None

    if origin is Literal and args:
        return args[0]

    if origin is list:
        return []
    if origin is dict:
        return {}
    if origin is tuple:
        return tuple()
    if origin is set:
        return set()

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            first_member = next(iter(annotation))
            return getattr(first_member, "value", first_member)
        if issubclass(annotation, bool):
            return False
        if issubclass(annotation, int):
            return 0
        if issubclass(annotation, float):
            return 0.0
        if issubclass(annotation, str):
            return "mock"

    return "mock"


class _MockLLMClient:
    """Basic mock client returning deterministic payloads for tests."""

    max_completion_tokens: int = 4096
    model: str = "mock-llm"

    def _build_payload(self, response_model: Type) -> Any:
        field_values = {}
        for field_name, field_info in getattr(response_model, "model_fields", {}).items():
            if field_info.default is not PydanticUndefined:
                field_values[field_name] = field_info.default
            elif getattr(field_info, "default_factory", None) is not None:
                field_values[field_name] = field_info.default_factory()
            else:
                field_values[field_name] = _coerce_default_value(field_info.annotation)
        try:
            return response_model(**field_values)
        except Exception:  # noqa: BLE001
            return response_model.model_construct(**field_values)

    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type
    ):
        return self._build_payload(response_model)

    def create_structured_output(self, text_input: str, system_prompt: str, response_model: Type):
        return self._build_payload(response_model)

    async def create_transcript(self, *args, **kwargs):
        return ""

    async def transcribe_image(self, *args, **kwargs):
        return {}


# Define an Enum for LLM Providers
class LLMProvider(Enum):
    """
    Define an Enum for identifying different LLM Providers.

    This Enum includes the following members:
    - OPENAI: Represents the OpenAI provider.
    - OLLAMA: Represents the Ollama provider.
    - ANTHROPIC: Represents the Anthropic provider.
    - CUSTOM: Represents a custom provider option.
    - GEMINI: Represents the Gemini provider.
    - MISTRAL: Represents the Mistral AI provider.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"
    GEMINI = "gemini"
    MISTRAL = "mistral"


def get_llm_client(raise_api_key_error: bool = True):
    """
    Get the LLM client based on the configuration using Enums.

    This function retrieves the configuration for the LLM provider and model, and
    initializes the appropriate LLM client adapter accordingly. It raises an
    LLMAPIKeyNotSetError if the LLM API key is not set for certain providers or if the provider
    is unsupported.

    Returns:
    --------

        An instance of the appropriate LLM client adapter based on the provider
        configuration.
    """
    if _should_use_mock_client():
        return _MockLLMClient()

    llm_config = get_llm_config()

    provider = LLMProvider(llm_config.llm_provider)

    # Check if max_token value is defined in liteLLM for given model
    # if not use value from cognee configuration
    from cognee.infrastructure.llm.utils import (
        get_model_max_completion_tokens,
    )  # imported here to avoid circular imports

    model_max_completion_tokens = get_model_max_completion_tokens(llm_config.llm_model)
    max_completion_tokens = (
        model_max_completion_tokens
        if model_max_completion_tokens
        else llm_config.llm_max_completion_tokens
    )

    if provider == LLMProvider.OPENAI:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
            OpenAIAdapter,
        )

        return OpenAIAdapter(
            api_key=llm_config.llm_api_key,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            model=llm_config.llm_model,
            transcription_model=llm_config.transcription_model,
            max_completion_tokens=max_completion_tokens,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            streaming=llm_config.llm_streaming,
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=llm_config.fallback_endpoint,
            fallback_model=llm_config.fallback_model,
        )

    elif provider == LLMProvider.OLLAMA:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
            GenericAPIAdapter,
        )

        return OllamaAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Ollama",
            max_completion_tokens=max_completion_tokens,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
        )

    elif provider == LLMProvider.ANTHROPIC:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter import (
            AnthropicAdapter,
        )

        return AnthropicAdapter(
            max_completion_tokens=max_completion_tokens,
            model=llm_config.llm_model,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
        )

    elif provider == LLMProvider.CUSTOM:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
            GenericAPIAdapter,
        )

        return GenericAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Custom",
            max_completion_tokens=max_completion_tokens,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=llm_config.fallback_endpoint,
            fallback_model=llm_config.fallback_model,
        )

    elif provider == LLMProvider.GEMINI:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.gemini.adapter import (
            GeminiAdapter,
        )

        return GeminiAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
        )

    elif provider == LLMProvider.MISTRAL:
        if llm_config.llm_api_key is None:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter import (
            MistralAdapter,
        )

        return MistralAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            endpoint=llm_config.llm_endpoint,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
        )

    else:
        raise UnsupportedLLMProviderError(provider)
