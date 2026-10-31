from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class DeepSeekChatOpenAI(ChatOpenAI):
    """Use DeepSeek's non-thinking tool-call dialect through the same model API."""

    # Forced tool selection is unsupported in thinking mode. Keeping it disabled
    # also avoids carrying provider-private reasoning through Agent checkpoints.
    extra_body: dict[str, Any] | None = Field(
        default_factory=lambda: {"thinking": {"type": "disabled"}}
    )

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel] | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        structured = super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )
        if (
            isinstance(schema, type)
            and issubclass(schema, BaseModel)
            and not include_raw
        ):
            # A successful HTTP response may omit the requested tool call. The
            # inherited parser then returns None rather than validating it.
            # Reject this inside the caller's ProviderExecutor operation so its
            # existing deadline and retry policy remain the sole owners.
            def validate_result(value: Any) -> BaseModel:
                return schema.model_validate(value)

            return structured | RunnableLambda(validate_result)
        return structured
