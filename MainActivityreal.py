from ultralytics import YOLO
import cv2
import socket

# ESP32 IP 및 포트 설정
esp_ip = '192.168.71.111'  # ESP32의 IP 주소를 확인하세요
esp_port = 80

# YOLOv8 모델 로드
model = YOLO("runs/detect/best.pt")  # Roboflow에서 학습한 모델을 불러옵니다.

# ESP32와 소켓 연결 시도
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((esp_ip, esp_port))
    print(f"Connected to ESP32 at {esp_ip}:{esp_port}")
except socket.error as e:
    print(f"Could not connect to ESP32: {e}")
    exit()

# 웹캠 신호 받기
video_signal = cv2.VideoCapture(0)

while True:
    # 웹캠 프레임 읽기
    ret, frame = video_signal.read()
    if not ret:
        break

    # YOLOv8을 사용한 객체 감지
    results = model(frame)

    # 결과 처리
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = box.conf[0]

            if label == "autochair" and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 경계상자 좌표
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 경계상자 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # 중심점 표시
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"({center_x},{center_y})", (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # ESP32로 데이터 전송
                try:
                    data = f"{label},{center_x},{center_y}\n"
                    client_socket.send(data.encode())
                except socket.error as e:
                    print(f"Failed to send data to ESP32: {e}")

    # 결과 프레임 표시
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 소켓 종료
video_signal.release()
cv2.destroyAllWindows()
client_socket.close()
