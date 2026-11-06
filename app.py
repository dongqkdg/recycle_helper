from pathlib import Path
import io
import os
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).parent
DEPLOY_VERSION = os.getenv("DEPLOY_VERSION", "개발환경")
app = FastAPI(title="분리수거 AI 분류기")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

GUIDES = {
    "플라스틱": "내용물을 비우고 가볍게 헹군 뒤 라벨과 뚜껑을 분리해 플라스틱으로 배출하세요.",
    "종이": "테이프·스프링·비닐을 제거하고 젖지 않게 접어 종이류로 배출하세요.",
    "캔": "내용물을 비우고 헹군 뒤 캔류로 배출하세요. 날카로운 부분은 눌러 주세요.",
    "유리": "내용물을 비우고 헹군 뒤 유리병류로 배출하세요. 깨진 유리는 별도 안전 포장하세요.",
    "일반쓰레기": "재활용이 어려운 오염물은 지자체 기준에 따라 종량제 봉투에 배출하세요.",
}

INDEX = {"플라스틱": ["bottle", "plastic", "container", "cup"], "종이": ["paper", "book", "envelope", "carton"], "캔": ["can", "tin", "bottle"], "유리": ["glass", "jar", "bottle"]}

def classify(image: Image.Image, filename: str = ""):
    """Use MobileNetV3 when torchvision is available; otherwise provide a transparent demo fallback."""
    try:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights).eval()
        with torch.inference_mode():
            scores = model(weights.transforms()(image).unsqueeze(0))[0].softmax(0)
        labels = weights.meta["categories"]
        hits = [(float(scores[i]), labels[i]) for i in scores.argsort(descending=True)[:8]]
        for category, words in INDEX.items():
            for score, label in hits:
                if any(word in label.lower() for word in words):
                    return category, round(min(0.98, max(0.51, score * 2.4)), 2), "MobileNetV3-Small (ImageNet)"
        return "일반쓰레기", round(max(.51, float(scores.max())), 2), "MobileNetV3-Small (ImageNet)"
    except (ImportError, ModuleNotFoundError):
        # Do not pretend that geometry alone is an image classifier.
        # A filename hint is useful for local smoke-tests, otherwise report unavailable.
        name = filename.lower()
        hints = {
            "플라스틱": ["플라스틱", "plastic", "pet"],
            "종이": ["종이", "paper", "cardboard", "carton"],
            "캔": ["캔", "can", "tin", "aluminum"],
            "유리": ["유리", "glass", "jar"],
            "일반쓰레기": ["일반", "trash", "garbage", "waste"],
        }
        for category, words in hints.items():
            if any(word in name for word in words):
                return category, 0.20, "파일명 기반 데모 추정(모델 미설치)"
        return "분류 불가", 0.0, "분류 모델 미설치"

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/api/classify")
async def api_classify(request: Request):
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    if "multipart/form-data" not in content_type:
        return {"ok": False, "error": "multipart/form-data 형식의 이미지 업로드가 필요합니다.", "version": DEPLOY_VERSION}
    boundary = content_type.split("boundary=", 1)[-1].encode()
    parts = body.split(b"--" + boundary)
    file_type, filename, data = None, "uploaded-image", b""
    for part in parts:
        if b"filename=" not in part:
            continue
        head, _, payload = part.partition(b"\r\n\r\n")
        file_type = next((line.split(b":", 1)[1].strip().decode() for line in head.split(b"\r\n") if line.lower().startswith(b"content-type:")), None)
        for line in head.split(b"\r\n"):
            if b"filename=" in line:
                filename = line.split(b"filename=", 1)[1].strip(b'\"; ' ).decode(errors="ignore")
        data = payload.rstrip(b"\r\n-")
        break
    if not file_type or not file_type.startswith("image/"):
        return {"ok": False, "error": "JPG, PNG, WEBP 등 이미지 파일만 업로드할 수 있습니다.", "version": DEPLOY_VERSION}
    if len(data) > 10 * 1024 * 1024:
        return {"ok": False, "error": "이미지는 10MB 이하만 업로드할 수 있습니다.", "version": DEPLOY_VERSION}
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        category, confidence, model = classify(image, filename)
        if category == "분류 불가":
            return {"ok": False, "error": "분류 모델이 설치되지 않았습니다. torch와 torchvision을 설치한 뒤 다시 실행하세요.", "model": model, "version": DEPLOY_VERSION}
        return {"ok": True, "category": category, "confidence": confidence, "guide": GUIDES[category], "model": model, "filename": filename, "version": DEPLOY_VERSION}
    except (UnidentifiedImageError, OSError):
        return {"ok": False, "error": "이미지 파일을 읽을 수 없습니다. 손상되지 않은 파일을 선택하세요.", "version": DEPLOY_VERSION}
