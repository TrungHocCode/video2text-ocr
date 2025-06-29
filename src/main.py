import time
from json_process import save_json_per_stock, extract_first_frame
from paddleocr import PaddleOCR
from logger_config import logger
import cv2
from excel_process import export_to_excel
def main():
    logger.info("Chương trình bắt đầu...")
    try:
        video_path = input("Nhập đường dẫn video: ")
        ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Không thể mở video.")
            return

        frame_delay = 0.4  # giây
        prev_region = None
        saved_bboxes = None
        stock_count = None
        first_frame_processed = False
        frame_id = 0

        frame_batch = []
        time_pos = 0.0
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
        logger.info(f"Tổng thời lượng video: {video_duration:.2f}s")

        while time_pos <= video_duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_pos * 1000)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Không thể đọc frame tại {time_pos:.2f}s")
                break

            frame_batch.append(frame)
            frame_id += 1
            time_pos += frame_delay

            # Nếu chưa xử lý frame đầu tiên
            if not first_frame_processed:
                first_result = extract_first_frame(ocr, frame)
                stock_count = first_result['stock_count']
                saved_bboxes = first_result['stocks']
                first_frame_processed = True
                
            # Nếu đủ 100 frame thì xử lý batch
            if len(frame_batch) == 100:
                logger.info(f"Xử lý batch {frame_id - 99} đến {frame_id}")
                for i, frame_in_batch in enumerate(frame_batch):
                    abs_id = frame_id - len(frame_batch) + i + 1
                    logger.info(f"Xử lý frame thứ {abs_id}")
                    prev_region = save_json_per_stock(
                        frame_in_batch,
                        ocr,
                        prev_regions=prev_region,
                        stock_count=stock_count,
                        output_dir="./output",
                        saved_bboxes=saved_bboxes,
                        frame_id=abs_id
                    )
                frame_batch.clear()  # giải phóng batch khỏi RAM
        print(frame_id)
        # Xử lý batch cuối nếu còn dư
        if frame_batch:
            logger.info(f"Xử lý batch cuối từ frame {frame_id - len(frame_batch) + 1} đến {frame_id}")
            for i, frame_in_batch in enumerate(frame_batch):
                abs_id = frame_id - len(frame_batch) + i + 1
                logger.info(f"Xử lý frame thứ {abs_id}")
                prev_region = save_json_per_stock(
                    frame_in_batch,
                    ocr,
                    prev_regions=prev_region,
                    stock_count=stock_count,
                    output_dir="./output",
                    saved_bboxes=saved_bboxes,
                    frame_id=abs_id
                )

        cap.release()
        elapsed = time.time() - start_time
        for i in range(stock_count):
            name=saved_bboxes[i]['stock_code']
            json_path=f"output/{name}.json"
            export_to_excel(json_path)        
        logger.info(f"Hoàn tất. Tổng thời gian xử lý: {elapsed:.2f} giây")

    except Exception as e:
        logger.exception("Đã xảy ra lỗi trong quá trình xử lý: %s", e)

if __name__ == "__main__":
    main()
