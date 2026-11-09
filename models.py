"""יצירת מודל ה-Embeddings ומודל השפה (במקום אחד, לפי config.py)."""
import os
from functools import lru_cache

import config as cfg


def _google_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("חסר GOOGLE_API_KEY בקובץ .env (מפתח חינמי: aistudio.google.com).")
    return key


def _make_gemini_embedding():
    """Embeddings דרך Gemini API. עטיפה קטנה כדי לא להסתמך על חבילה נוספת."""
    import time
    from typing import Any

    from llama_index.core.base.embeddings.base import BaseEmbedding
    from pydantic import PrivateAttr

    class GeminiEmbedding(BaseEmbedding):
        _client: Any = PrivateAttr()
        _model: str = PrivateAttr()

        def __init__(self, api_key: str, model_name: str, **kwargs):
            super().__init__(model_name=model_name, **kwargs)
            from google import genai

            self._client = genai.Client(api_key=api_key)
            self._model = model_name

        @classmethod
        def class_name(cls) -> str:
            return "GeminiEmbedding"

        def _embed(self, texts: list[str], task: str) -> list[list[float]]:
            from google.genai import types

            for attempt in range(4):  # ניסיון חוזר קצר במקרה של מגבלת קצב (429)
                try:
                    res = self._client.models.embed_content(
                        model=self._model,
                        contents=texts,
                        config=types.EmbedContentConfig(task_type=task),
                    )
                    return [list(e.values) for e in res.embeddings]
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(10 * (attempt + 1))

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._embed([query], "RETRIEVAL_QUERY")[0]

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._embed([text], "RETRIEVAL_DOCUMENT")[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self._embed(texts, "RETRIEVAL_DOCUMENT")

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

    return GeminiEmbedding(api_key=_google_key(), model_name=cfg.EMBED_MODEL, embed_batch_size=16)


@lru_cache(maxsize=1)
def get_embed_model():
    if cfg.EMBED_PROVIDER == "gemini":
        return _make_gemini_embedding()

    if cfg.EMBED_PROVIDER == "local":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(
            model_name=cfg.EMBED_MODEL,
            query_instruction=cfg.EMBED_QUERY_PREFIX or None,
            text_instruction=cfg.EMBED_TEXT_PREFIX or None,
        )

    raise RuntimeError(f"EMBED_PROVIDER לא מוכר: {cfg.EMBED_PROVIDER} (gemini או local)")


@lru_cache(maxsize=1)
def get_llm():
    if cfg.LLM_PROVIDER == "gemini":
        from llama_index.llms.google_genai import GoogleGenAI

        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("חסר GOOGLE_API_KEY בקובץ .env (מפתח חינמי: aistudio.google.com).")
        # בלי max_tokens: במודלי Gemini "החשיבה" נספרת מול המגבלה ועלולה לקצץ את התשובה
        return GoogleGenAI(model=cfg.LLM_MODEL, api_key=key, temperature=cfg.LLM_TEMPERATURE)

    if cfg.LLM_PROVIDER == "anthropic":
        from llama_index.llms.anthropic import Anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("חסר משתנה הסביבה ANTHROPIC_API_KEY (ראו .env.example).")
        return Anthropic(
            model=cfg.LLM_MODEL,
            temperature=cfg.LLM_TEMPERATURE,
            max_tokens=cfg.LLM_MAX_TOKENS,
        )

    raise RuntimeError(f"LLM_PROVIDER לא מוכר: {cfg.LLM_PROVIDER} (gemini או anthropic)")
