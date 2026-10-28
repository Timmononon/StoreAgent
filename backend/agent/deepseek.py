from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
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
        # The inherited tool parser still validates Pydantic schemas. There is no
        # request-time protocol probing, retry, or unvalidated-text fallback.
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )
