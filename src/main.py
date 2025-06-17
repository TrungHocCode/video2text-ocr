import time
from get_frames import extract_frames
from json_process import save_json_per_stock
from paddleocr import PaddleOCR
from logger_config import logger
from excel_process import export_to_excel
# def main():
#     logger.info("Chương trình bắt đầu...")
#     try:
#         video_path = input("Nhập đường dẫn video: ")
#         user_stock_require = int(input("Nhập số lượng mã cần lấy: "))

#         ocr = PaddleOCR(use_angle_cls=True, lang="vi",show_log=False)
#         start_time = time.time()

#         frames_list = extract_frames(video_path, 0.4, "./frames")
#         prev_region = None

#         for idx, frame in enumerate(frames_list):
#             logger.info(f"Xử lý frame thứ {idx + 1}/{len(frames_list)}")
#             prev_region = save_json_per_stock(
#                 frame,
#                 ocr,
#                 prev_regions=prev_region,
#                 stock_count=3, #Trong trường hợp có nhiều hơn 3 mã thì thay stock_count bằng giá trị khác
#                 output_dir="./output",
#                 stock_require=user_stock_require
#             )

#         elapsed = time.time() - start_time
#         logger.info(f"Hoàn tất. Tổng thời gian xử lý: {elapsed:.2f} giây")

#     except Exception as e:
#         logger.exception("Đã xảy ra lỗi trong quá trình xử lý: %s", e)
def main():
    logger.info("Chương trình bắt đầu...")
    try:
        video_path = input("Nhập đường dẫn video: ")
        user_stock_require = int(input("Nhập số lượng mã cần lấy: "))

        ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
        start_time = time.time()

        frames_list = extract_frames(video_path, 0.4, "./frames")
        prev_region = None
        saved_bboxes = None

        for idx, frame in enumerate(frames_list):
            logger.info(f"Xử lý frame thứ {idx + 1}/{len(frames_list)}")
            prev_region, saved_bboxes = save_json_per_stock(
                frame,
                ocr,
                prev_regions=prev_region,
                stock_count=3,
                output_dir="./output",
                stock_require=user_stock_require,
                saved_bboxes=saved_bboxes,frame_id=idx
            )
        elapsed = time.time() - start_time
        for i in range(3):
            name=saved_bboxes[i]['stock_code']
            json_path=f"output/{name}.json"
            export_to_excel(json_path)
        logger.info(f"Hoàn tất. Tổng thời gian xử lý: {elapsed:.2f} giây")

    except Exception as e:
        logger.exception("Đã xảy ra lỗi trong quá trình xử lý: %s", e)
if __name__ == "__main__":
    main()