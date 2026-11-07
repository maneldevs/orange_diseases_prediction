import base64
import io
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from predict import Prediction, predict as model_predict

MAX_BYTES = 5 * 1024 * 1024          # 5 MB
CHUNK = 1024 * 1024
TYPES = {"image/jpeg", "image/png"}
Image.MAX_IMAGE_PIXELS = 50_000_000
MODEL_PATH = 'models/orange_diseases_model_1.onnx'


app = FastAPI(title="Orange Diseases Prediction")
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

async def read_file(file: UploadFile) -> bytes | None:
    buffer = io.BytesIO()
    while chunk := await file.read(CHUNK):
        buffer.write(chunk)
        if buffer.tell() > MAX_BYTES:
            return None
    return buffer.getvalue()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"prediction": None, "error": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        if file.content_type not in TYPES:
            return templates.TemplateResponse(request, 400, "index.html", {"prediction": None, "error": 'Format not supported'})
        if file.size is not None and file.size > MAX_BYTES:
            return templates.TemplateResponse(request, 413, "index.html", {"prediction": None, "error": 'Image max size: 5 MB'})
        content = await read_file(file)
        if content is None:
            return templates.TemplateResponse(request, 413, "index.html", {"prediction": None, "error": 'Image max size: 5 MB'})
    except Exception:
        return templates.TemplateResponse(request, "index.html", {"prediction": None, "error": 'Error loading file'})
    finally:
        await file.close()
    prediction = model_predict(image_input=content, model_path=MODEL_PATH)
    base64_encoded = base64.b64encode(content).decode("utf-8")
    preview_url = f"data:{file.content_type};base64,{base64_encoded}"
    return templates.TemplateResponse(request, "index.html", {"prediction": prediction, "preview": preview_url, "error": None})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)