import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

# EAR 계산 함수
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 얼굴 탐지기와 얼굴 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')  # 파일 경로 수정

# 눈 랜드마크 인덱스 설정
(lStart, lEnd) = (36, 41)  # 왼쪽 눈
(rStart, rEnd) = (42, 47)  # 오른쪽 눈

# EAR 임계값 및 프레임 카운터 설정
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15
frame_counter = 0

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠이 성공적으로 열렸습니다. 실시간 스트림을 시작합니다.")

# Matplotlib 초기 설정
plt.ion()  # Interactive mode on
fig, ax = plt.subplots(figsize=(10, 6))

# 실시간 비디오 스트림 처리
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        # 얼굴 랜드마크 예측
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # 눈 좌표 추출 및 EAR 계산
        leftEye = shape[lStart:lEnd + 1]
        rightEye = shape[rStart:rEnd + 1]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0  # 양쪽 눈의 평균 EAR

        # 눈 감지 마커 표시 (폴리곤)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # 초록색
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # 초록색

        # EAR 값 기준으로 졸음 탐지
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTION", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 빨간색
        else:
            frame_counter = 0

        # EAR 값 화면에 표시
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Matplotlib로 프레임 표시
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR, Matplotlib은 RGB 사용
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis('off')
    plt.pause(0.01)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
plt.close()
print("프로그램 종료")
