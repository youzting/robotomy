import torch
import cv2

# YOLOv5 모델 불러오기 (yolov5s: 빠르고 가벼움)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold (기본값은 0.25)

# 웹캠 열기 (0: 기본 카메라)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5에 이미지 전달 → 감지 결과 반환
    results = model(frame)

    # 결과 시각화 이미지 얻기
    annotated_frame = results.render()[0]

    # 탐지된 물체 이름 리스트
    names = results.pandas().xyxy[0]['name'].tolist()
    print("인식된 객체:", names)

    # 시각화된 결과 화면에 출력
    cv2.imshow("YOLOv5 실시간 인식", annotated_frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
