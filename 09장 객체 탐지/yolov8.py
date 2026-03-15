print("starting...")
from ultralytics import YOLO

model = YOLO("../models/yolov8m-pose.pt")


import torch

def predict(frame, iou=0.7, conf=0.25):
    results = model(
        source=frame,
        device="0" if torch.cuda.is_available() else "cpu",
        iou=0.7,
        conf=0.25,
        verbose=False,
    )
    result = results[0]
    return result

def draw_boxes(result, frame):
    for boxes in result.boxes:
        x1, y1, x2, y2, score, classes = boxes.data.squeeze().cpu().numpy()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    return frame


from ultralytics.yolo.utils.plotting import Annotator

def draw_keypoints(result, frame):
    annotator = Annotator(frame, line_width=1)
    for kps in result.keypoints:
        kps = kps.data.squeeze()
        annotator.kpts(kps)
        
        nkps = kps.cpu().numpy()
        # nkps[:,2] = 1
        # annotator.kpts(nkps)
        for idx, (x, y, score) in enumerate(nkps):
            if score > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        
    return frame


import cv2

print("video loading and displaying...")
capture = cv2.VideoCapture("../datasets/woman.mp4")
while cv2.waitKey(10) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    ret, frame = capture.read()
    result = predict(frame)
    frame = draw_boxes(result, frame)
    frame = draw_keypoints(result, frame)
    cv2.imshow("VideoFrame", frame)

capture.release()
cv2.destroyAllWindows()
