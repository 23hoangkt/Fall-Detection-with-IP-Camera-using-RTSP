# Tổng quan hệ thống

<p align="center">
  <b>Fall Detection</b><br>
  <img src="Screenshot 2024-09-20 002229.png" alt="Mô tả hình ảnh 1" width="400"/>
</p>

# Dowload Models tại 
https://drive.google.com/drive/u/0/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO

# Chạy với file main
```python main.py```
# Test RTSP với camera IP
kết nối với camera và lấy địa chỉ IP của camera
```python rtsp.py```

# Chạy mô hình kết hợp với SMS, Call
đăng ký tài khoản Twilo, nhập thông tin
```python fall_detection.py```

# Sự kết hợp giữa 3 models Yolo-Tiny V3, PSSE, STGCN
bạn có thể train lại mô hình yolov3-tiny và lấy 1 file weight mới 
# Phát hiện lửa nhỏ với camera IP với YoLoV8
bộ dữ liệu lấy từ RoboFlow , train thông qua GG COLAB ```python fire.py```
