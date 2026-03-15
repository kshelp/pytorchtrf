import cv2
import numpy as np
import re
from ultralytics import YOLO
import easyocr

# =========================
# 모델 로드
# =========================
# https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/blob/main/license_plate_detector.pt
model = YOLO("../models/license_plate_detector.pt")
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# =========================
# 이미지 로드
# =========================
img = cv2.imread("../datasets/car001.jpg")
img = np.ascontiguousarray(img)

# =========================
# YOLO 추론
# =========================
results = model(img)

# =========================
# 번호판 처리
# =========================
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 번호판 Crop
        plate_img = img[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue

        # =========================
        # OCR 전처리 (한글 번호판 핵심)
        # =========================
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # =========================
        # OCR
        # =========================
        ocr_result = reader.readtext(
            thresh,
            detail=1,
            paragraph=False,
            text_threshold=0.6,
            low_text=0.3
        )

        # =========================
        # 한글 번호판 후처리
        # =========================
        plate_text = ""
        for (_, text, conf) in ocr_result:
            if conf > 0.4:
                # 숫자 + 한글만 허용
                clean_text = re.sub(r'[^0-9가-힣]', '', text)
                plate_text += clean_text

        # 번호판 형식 보정 (예: 12가3456)
        plate_text = plate_text.strip()

        # =========================
        # 시각화
        # =========================
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            plate_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

# =========================
# 결과 출력
# =========================
#cv2.imshow("License Plate Recognition (KR)", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
print("차량번호="+plate_text)