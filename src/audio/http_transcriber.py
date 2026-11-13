"""Transcrição via API HTTP de terceiros (Knowlez STT) — alternativa ao Azure Speech.

Contrato observado: POST com header `x-api-key` e JSON {"audio_base64", "language"};
resposta {"text", "language", "duration_seconds", "segments": [{start, end, text}]}.
A chave vem do `.env` (STT_API_KEY); nunca a versione no Git.
"""
from __future__ import annotations

import base64

import requests

from ..config import settings


class HttpSttTranscriber:
    def __init__(self, url: str | None = None, key: str | None = None, language: str = "pt-BR"):
        self.url = url or settings.stt_api_url
        self.key = key or settings.stt_api_key
        self.language = language

    def transcribe_full(self, wav_path: str, timeout_s: int = 120) -> dict:
        if not (self.url and self.key):
            raise RuntimeError("STT_API_URL / STT_API_KEY não configuradas")
        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        resp = requests.post(self.url, headers={"x-api-key": self.key},
                             json={"audio_base64": audio_b64, "language": self.language},
                             timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()

    def transcribe(self, wav_path: str, timeout_s: int = 120) -> str:
        return (self.transcribe_full(wav_path, timeout_s).get("text") or "").strip()
