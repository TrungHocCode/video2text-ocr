from logger_config import logger
import cv2
import numpy as np
import os
def extract_frames(video_path, delay=0.4,save_dir=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở video!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    time_points = np.arange(0, duration_sec, delay)

    all_frames = []
    count = 0

    # if save_dir:
    #     os.makedirs(save_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    for t in time_points:
        frame_id = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        all_frames.append(frame)
        count += 1

        # Nếu muốn lưu frames đã trích xuất từ video unconmment đoạn code dưới và tạo sẵn thư mục để chứa frames rồi đưa voà tham số save_dir của hàm 
        # if save_dir:
        #     frame_filename = os.path.join(save_dir, f"frame_{count:04d}.jpg") 
        #     cv2.imwrite(frame_filename, frame)

        logger.info(f"Đã trích xuất và lưu frame {count}")

    cap.release()
    return all_frames

