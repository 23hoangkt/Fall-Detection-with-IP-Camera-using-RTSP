import cv2
import time
from twilio.rest import Client
from ultralytics import YOLO

account_sid = ''  # Thay bằng Account SID của bạn
auth_token = ''  # Thay bằng Auth Token của bạn
twilio_phone_number = ''  # Thay bằng số điện thoại Twilio của bạn
target_phone_number = ''  # Thay bằng số điện thoại nhận tin nhắn

# Khởi tạo Twilio Client
twilio_client = Client(account_sid, auth_token)

def send_sms(body):
    try:
        message = twilio_client.messages.create(
            body=body,
            from_=twilio_phone_number,
            to=target_phone_number
        )
        print(f"Tin nhắn đã được gửi thành công. SID: {message.sid}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Tải mô hình YOLOv8
model = YOLO("best (2).pt")

# Khởi tạo webcam (hoặc video file)
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định. Bạn có thể thay đổi thành đường dẫn tới file video nếu cần

# Biến để theo dõi thời gian lửa xuất hiện
fire_start_time = None
fire_detected = False
fire_sms_sent = False

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Thực hiện dự đoán
    results = model.predict(frame)

    # Kiểm tra nếu lửa được phát hiện trong frame hiện tại
    fire_in_frame = False
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            if label == "fire":  # Giả sử "fire" là nhãn cho lửa trong dataset của bạn
                fire_in_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Cập nhật thời gian phát hiện lửa
    if fire_in_frame:
        if fire_start_time is None:
            fire_start_time = time.time()
        elif time.time() - fire_start_time > 3:
            fire_detected = True
    else:
        fire_start_time = None
        fire_detected = False
        fire_sms_sent = False  # Đặt lại cờ khi lửa biến mất

    # In ra thông báo và gửi SMS nếu cần
    if fire_detected and not fire_sms_sent:
        send_sms("Có lửa!")
        fire_sms_sent = True  # Đặt cờ để không gửi tin nhắn lặp lại
    elif not fire_detected:
        print("No fire")

    # # Hiển thị frame
    # cv2.imshow("Fire Detection", frame)

    # # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
