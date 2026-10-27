"""VLM backends for floor selection: ``local`` (in-process HF), ``served``
(any OpenAI-compatible chat endpoint) and ``openai`` (Responses API).

Import-light: nothing here loads ``torch``, ``transformers`` or ``openai``.
"""
from .client import (
    BACKENDS,
    VlmClient,
    VlmImage,
    VlmRequest,
    VlmResponse,
    VlmUnavailableError,
    vlm_client,
)

__all__ = [
    "BACKENDS", "VlmClient", "VlmImage", "VlmRequest", "VlmResponse",
    "VlmUnavailableError", "vlm_client",
]
