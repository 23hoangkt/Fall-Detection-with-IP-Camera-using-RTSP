import threading
import ffmpeg
import numpy as np
import cv2
import queue
import time
import torch
import argparse
import os

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from twilio.rest import Client

# Cấu hình Twilio
TWILIO_ACCOUNT_SID = ''
TWILIO_AUTH_TOKEN = ''    
TWILIO_PHONE_NUMBER = ''
TO_PHONE_NUMBER = ''  # Số điện thoại bạn muốn gửi tin nhắn đến

# Khởi tạo Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms(message):
    """Gửi tin nhắn SMS qua Twilio."""
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=TO_PHONE_NUMBER
        )
        print("Tin nhắn đã được gửi: ", message)
    except Exception as e:
        print("Không thể gửi tin nhắn: ", e)

# Tạo hàng đợi toàn cục để lưu trữ khung hình
frame_queue = queue.Queue(maxsize=10)

# Đoạn code 1: Xử lý video từ RTSP và lưu khung hình vào hàng đợi
def video_processing(cam_source):
    args = {"rtsp_transport": "udp"}
    probe = ffmpeg.probe(cam_source)
    cap_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
    print("fps: {}".format(cap_info["r_frame_rate"]))
    width = cap_info["width"]
    height = cap_info["height"]
    up, down = str(cap_info["r_frame_rate"]).split("/")
    fps = eval(up) / eval(down)
    print("fps: {}".format(fps))
    
    process1 = (
        ffmpeg.input(cam_source, **args)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )
    
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        frame = cv2.resize(in_frame, (384, 384))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not frame_queue.full():
            frame_queue.put(frame)

        cv2.imshow("ffmpeg", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    process1.kill()

# Đoạn code 2: Xử lý video từ hàng đợi
def video_detection():
    def preproc(image):
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def kpt2bbox(kpt, ex=20):
        return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                         kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default='')
    par.add_argument('--detection_input_size', type=int, default=384)
    par.add_argument('--pose_input_size', type=str, default='224x160')
    par.add_argument('--pose_backbone', type=str, default='resnet50')
    par.add_argument('--show_detected', default=False, action='store_true')
    par.add_argument('--show_skeleton', default=True, action='store_true')
    par.add_argument('--save_out', type=str, default='')
    par.add_argument('--device', type=str, default='cuda')
    args = par.parse_args()

    device = args.device
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    fall_detected = False  # Cờ để kiểm tra trạng thái ngã
    fall_cooldown = 100  # Thời gian chờ (giây) trước khi gửi lại tin nhắn
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            image = frame.copy()

            detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

            tracker.predict()
            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det

            detections = []
            if detected is not None:
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]

                if args.show_detected:
                    for bb in detected[:, 0:5]:
                        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

            tracker.update(detections)

            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                action = 'pending..'
                clr = (0, 255, 0)
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    if action_name == 'Fall Down':
                        clr = (255, 0, 0)
                        current_time = time.time()
                        # if not fall_detected:  # Kiểm tra xem tin nhắn đã được gửi chưa
                        if not fall_detected or (current_time - last_fall_time) > fall_cooldown:
                            send_sms('Cảnh báo: Có người ngã ở phòng khách')
                            fall_detected = True
                            last_fall_time = current_time
                    elif action_name == 'Lying Down':
                        clr = (255, 200, 0)
                        fall_detected = False

                if track.time_since_update == 0:
                    if args.show_skeleton:
                        frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, (255, 0, 0), 2)
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                        0.4, clr, 1)

            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = frame[:, :, ::-1]
            fps_time = time.time()

            if outvid:
                writer.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if outvid:
        writer.release()
    cv2.destroyAllWindows()

# Khởi động các luồng
if __name__ == "__main__":
    cam_source1 = ""
    video_thread = threading.Thread(target=video_processing, args=(cam_source1,))
    video_thread.start()

    detection_thread = threading.Thread(target=video_detection)
    detection_thread.start()

    video_thread.join()
    detection_thread.join()
