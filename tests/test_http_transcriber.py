from unittest.mock import patch, MagicMock

import pytest

from src.audio.http_transcriber import HttpSttTranscriber


def test_transcribe_envia_base64_e_header(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    resp = MagicMock(); resp.json.return_value = {"text": " dor no peito "}
    with patch("src.audio.http_transcriber.requests.post", return_value=resp) as post:
        out = HttpSttTranscriber("http://x", "k").transcribe(str(wav))
    assert out == "dor no peito"
    kw = post.call_args.kwargs
    assert kw["headers"] == {"x-api-key": "k"}
    assert kw["json"]["language"] == "pt-BR" and kw["json"]["audio_base64"]


def test_sem_credenciais_levanta_erro(tmp_path):
    with pytest.raises(RuntimeError):
        HttpSttTranscriber("", "").transcribe(str(tmp_path / "a.wav"))
