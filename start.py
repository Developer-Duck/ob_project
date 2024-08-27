import cv2
from ultralytics import YOLO
import pyautogui
import numpy as np

# 학습한 YOLO 모델 경로를 지정합니다.
model_path = "C:/Users/xxxfl/OneDrive/바탕 화면/동아리/bl_object_project/runs/detect/train81/weights/best.pt"

# 학습한 YOLO 모델 로드
model = YOLO(model_path)

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 기본 웹캠 사용


#   
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 높이 설정

# 클릭을 방지하기 위한 플래그
click_triggered = False

while True:
    # 프레임 캡처
    ret, frame = cap.read()




    if not ret:
        break

    # 객체 감지 수행
    results = model(frame)

    # 감지된 객체의 정보를 가져옵니다.
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
    confidences = results[0].boxes.conf.cpu().numpy()  # 신뢰도
    class_ids = results[0].boxes.cls.cpu().numpy()  # 클래스 ID

    if len(confidences) == 0:
        # 감지된 객체가 없는 경우
        cv2.imshow('YOLO Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 손가락 감지를 위한 클래스 ID (예: 0으로 설정)
    finger_class_id = 0  # 실제 모델에서 손가락의 클래스 ID로 변경 필요

    # 가장 높은 신뢰도를 가진 손가락 바운딩 박스 찾기
    max_confidence_index = np.argmax(confidences)
    max_confidence_class_id = int(class_ids[max_confidence_index])

    if max_confidence_class_id == finger_class_id:
        # 바운딩 박스 좌표
        x1, y1, x2, y2 = boxes[max_confidence_index]

        # 프레임 크기와 웹캠의 해상도에 따라 마우스 좌표 변환
        frame_height, frame_width = frame.shape[:2]
        screen_width, screen_height = pyautogui.size()

        # 손가락 중심 좌표 계산
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 마우스 좌표 변환
        mouse_x = int(center_x / frame_width * screen_width)
        mouse_y = int(center_y / frame_height * screen_height)

        # 마우스 이동
        pyautogui.moveTo(mouse_x, mouse_y)

        # 손가락 바닥 부분이 화면 하단에 가까운지 확인
        if y2 > frame_height * 0.9:  # y2가 프레임 높이의 90% 이상인 경우
            if not click_triggered:
                print("Click triggered!")  # 디버깅용 로그 출력
                pyautogui.click()
                click_triggered = True
        else:
            click_triggered = False
    else:
        click_triggered = False

    # 프레임에 결과 렌더링
    annotated_frame = results[0].plot()  # plot()을 사용하여 프레임에 결과를 그림

    # 결과 프레임 표시
    cv2.imshow('YOLO Object Detection', annotated_frame)

    # 'q'를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
