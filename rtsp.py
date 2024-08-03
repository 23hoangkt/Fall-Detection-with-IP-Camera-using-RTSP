import ffmpeg
import numpy as np
import cv2


def main(cam_source):
    args = {"rtsp_transport": "udp"}  # 添加参数
    probe = ffmpeg.probe(cam_source)
    cap_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
    print("fps: {}".format(cap_info["r_frame_rate"]))
    width = cap_info["width"]  # 获取视频流的宽度
    height = cap_info["height"]  # 获取视频流的高度
    up, down = str(cap_info["r_frame_rate"]).split("/")
    fps = eval(up) / eval(down)
    print("fps: {}".format(fps))  # 读取可能会出错错误
    process1 = (
        ffmpeg.input(cam_source, **args)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )
    while True:
        in_bytes = process1.stdout.read(width * height * 3)  # 读取图片
        if not in_bytes:
            break
        # 转成ndarray
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        frame = cv2.resize(in_frame, (384, 384))  # 改变图片尺寸
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转成BGR
        cv2.imshow("ffmpeg", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    process1.kill()  # 关闭


if __name__ == "__main__":
    cam_source = ""
    main(cam_source)




