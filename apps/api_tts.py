"""
VieNeu-TTS REST API — All-in-one (1 process duy nhất).

Load model trực tiếp trên GPU, nhận text trả audio WAV.
Không cần chạy LMDeploy server riêng.

Sử dụng:
    python3 apps/api_tts.py

Test:
    curl -o output.wav "http://<server-ip>:8002/tts?text=xin+chào+bạn"

Cấu hình cố định:
    - Model: pnnbao-ump/VieNeu-TTS-v2 (GPU, LMDeploy pipeline)
    - Codec: neuphonic/distill-neucodec (CUDA)
    - Giọng mặc định: Ly (nữ miền Bắc)
"""

import io
import wave
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from vieneu import Vieneu

# ==========================================
# CONFIG
# ==========================================
MODEL_REPO = "pnnbao-ump/VieNeu-TTS-v2"
CODEC_REPO = "neuphonic/distill-neucodec"
CODEC_DEVICE = "cuda"
MEMORY_UTIL = 0.3
DEFAULT_VOICE = "Ly"
HOST = "0.0.0.0"
PORT = 8014

# ==========================================
# INIT
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("VieNeu.API")

app = FastAPI(
    title="VieNeu-TTS API",
    description="Text-to-Speech API — Nhận text, trả audio WAV hoàn chỉnh",
    version="1.0.0"
)

# Load TTS engine (fast mode — load model trực tiếp trên GPU)
logger.info("🚀 Khởi tạo VieNeu-TTS (GPU mode)...")
logger.info(f"   Model: {MODEL_REPO}")
logger.info(f"   Codec: {CODEC_REPO} on {CODEC_DEVICE}")
logger.info(f"   Memory util: {MEMORY_UTIL}")
logger.info(f"   Giọng mặc định: {DEFAULT_VOICE}")

tts = Vieneu(
    mode="fast",
    backbone_repo=MODEL_REPO,
    backbone_device="cuda",
    codec_repo=CODEC_REPO,
    codec_device=CODEC_DEVICE,
    memory_util=MEMORY_UTIL,
)

# Load giọng mặc định
default_voice_data = tts.get_preset_voice(DEFAULT_VOICE)
logger.info(f"✅ Sẵn sàng! Giọng mặc định: {DEFAULT_VOICE}")


# ==========================================
# HELPERS
# ==========================================
def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Chuyển numpy audio array thành WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ==========================================
# API ENDPOINTS
# ==========================================
class TTSRequest(BaseModel):
    text: str = Field(..., description="Văn bản cần chuyển thành giọng nói")
    voice: Optional[str] = Field(default=None, description="Voice ID (mặc định: Ly)")


@app.get("/tts")
def tts_get(text: str, voice: Optional[str] = None):
    """
    GET /tts?text=...&voice=...

    Sinh audio từ text, trả về file WAV hoàn chỉnh.
    """
    return _synthesize(text, voice)


@app.post("/tts")
def tts_post(req: TTSRequest):
    """
    POST /tts

    Body JSON: {"text": "...", "voice": "..."}
    Sinh audio từ text, trả về file WAV hoàn chỉnh.
    """
    return _synthesize(req.text, req.voice)


@app.get("/voices")
def list_voices():
    """Danh sách giọng nói có sẵn."""
    voices = tts.list_preset_voices()
    return [{"id": v[1], "name": v[0]} for v in voices]


@app.get("/health")
def health_check():
    """Kiểm tra trạng thái server."""
    return {
        "status": "ok",
        "model": MODEL_REPO,
        "codec": CODEC_REPO,
        "default_voice": DEFAULT_VOICE,
    }


# ==========================================
# CORE LOGIC
# ==========================================
def _synthesize(text: str, voice: Optional[str] = None) -> Response:
    """Xử lý sinh audio và trả WAV response."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Thiếu text")

    text = text.strip()
    logger.info(f"🎤 Sinh audio: '{text[:50]}...' | voice={voice or DEFAULT_VOICE}")

    try:
        # Chọn giọng
        if voice:
            try:
                voice_data = tts.get_preset_voice(voice)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice '{voice}' không tồn tại. Dùng GET /voices để xem danh sách."
                )
        else:
            voice_data = default_voice_data

        # Sinh audio
        audio = tts.infer(text=text, voice=voice_data)

        if audio is None or len(audio) == 0:
            raise HTTPException(status_code=500, detail="Không sinh được audio")

        # Chuyển thành WAV
        wav_bytes = audio_to_wav_bytes(audio)
        logger.info(f"✅ Hoàn tất: {len(audio)} samples ({len(audio)/24000:.2f}s)")

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    logger.info(f"🌍 API running at http://{HOST}:{PORT}")
    logger.info(f"📖 Docs: http://{HOST}:{PORT}/docs")
    logger.info(f"")
    logger.info(f"Ví dụ:")
    logger.info(f'  curl -o output.wav "http://localhost:{PORT}/tts?text=chúng+ta+là+siêu+nhân+gao"')
    uvicorn.run(app, host=HOST, port=PORT)
