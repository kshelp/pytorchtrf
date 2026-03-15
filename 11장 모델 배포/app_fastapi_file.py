import io
import torch
import logging
from PIL import Image
from torch.nn import functional as F
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

# 로깅 설정 (에러 추적용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VGG16Model:
    def __init__(self, weight_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48235, 0.45882, 0.40784],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # 모델 로드 (weights=None으로 경고 방지 및 로컬 가중치 로드)
        self.model = models.vgg16(weights=None, num_classes=2).to(self.device)
        try:
            self.model.load_state_dict(
                torch.load(weight_path, map_location=self.device)
            )
            self.model.eval()
            logger.info(f"Model loaded successfully from {weight_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def preprocessing(self, image_bytes: bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # transform 결과에 .to(self.device)를 붙여 모델과 장치를 맞춤
            return self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")

    @torch.no_grad()
    def predict(self, image_bytes: bytes):
        # 1. 전처리
        x = self.preprocessing(image_bytes)
        
        # 2. 추론
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)

        # 3. 결과 해석
        idx = int(probs.argmax(dim=1).item())
        return {
            "label": "개" if idx == 1 else "고양이",
            "score": float(probs[0, idx].item()),
        }

app = FastAPI()

# 모델 경로 확인 필요
try:
    model = VGG16Model("../models/VGG16.pt")
except Exception:
    model = None # 서버 시작 시 모델 로드 실패 대응

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # 파일 읽기
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file provided")

        # 예측 실행
        result = model.predict(image_bytes)
        return result

    except ValueError as ve:
        # 이미지 변환 실패 등 값 관련 에러
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # 예측 도중 발생하는 예상치 못한 에러 로그 출력
        logger.error(f"Prediction error: {str(e)}")
        # 유저에게는 bytes가 포함되지 않은 깔끔한 텍스트만 전달
        raise HTTPException(
            status_code=500, 
            detail="An error occurred during prediction. Check the server logs."
        )

if __name__ == "__main__":
    # 파일명이 app_fastapi.py가 아닐 수도 있으므로 모듈 경로 확인
    uvicorn.run(app, host="0.0.0.0", port=8000)