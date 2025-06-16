import json
from datetime import datetime
import cv2
import numpy as np
from ocr_process import split_frame_by_stock_count, classify_ocr_regions,extract_stock_codes_for_filenames
import pandas as pd
def is_frame_changed(prev_frame, current_frame, threshold=1):
    if prev_frame is None:
        return True
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

def extract_json_from_ocr(ocr_result):
    trade_result=ocr_result['matched_trades']
    order_result=ocr_result['order_book']
    ref_time=datetime.now().strftime("%H:%M:%S")
    result_json={
        ref_time:{
            "Khớp" :{},
            "Giá"  :{},
            "KL"   :{},
            "M/B"  :{},
            "zone" :[]
        }
    }
    for idx, line in enumerate(trade_result[:2]):
        parts=line["text"].split()
        if len(parts) >= 5:
            lan = f"Lần {idx + 1}"
            time, price, diff, volume, mb = parts[:5]
            result_json[ref_time]["Khớp"][lan] = time
            result_json[ref_time]["Giá"][lan] = price
            result_json[ref_time]["KL"][lan] = volume
            result_json[ref_time]["M/B"][lan] = mb

    for idx, order in enumerate(order_result[:3]):
        parts = order["text"].split()
        if len(parts) >= 4:
            zone = {
                f"KL_mua {idx + 1}": parts[0],
                f"Gia_mua {idx + 1}": parts[1],
                f"Gia_ban {idx + 1}": parts[2],
                f"KL_ban {idx + 1}": parts[3]
            }
            result_json[ref_time]["zone"].append(zone)

    return result_json
def frame_changed(prev_region, curr_region, threshold=25, change_ratio=0.001):
    if prev_region is None:
        return True
    diff = cv2.absdiff(prev_region, curr_region)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    
    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    ratio = changed_pixels / total_pixels
    return ratio > change_ratio

def save_json_per_stock(frame, ocr,prev_regions=None, stock_count=3, output_dir="./",stock_require=1):
    stock_regions = split_frame_by_stock_count(frame, stock_count)

    if prev_regions is None:
        prev_regions = [None] * stock_count

    for idx, region in enumerate(stock_regions):
        if idx == stock_require:
            break
        # Kiểm tra thay đổi vùng
        if not frame_changed(prev_regions[idx], region,):
            continue  # Bỏ qua nếu không thay đổi
        
        
        # Cập nhật vùng trước
        prev_regions[idx] = region.copy()

        # OCR và xử lý JSON
        ocr_result = ocr.ocr(region, cls=False)
        result = classify_ocr_regions(ocr_result)

        # Trích xuất mã chứng khoán
        stock_codes = extract_stock_codes_for_filenames(result['stocks'])
        if not stock_codes:
            continue  # Bỏ qua nếu không xác định được mã

        stock_code = stock_codes[0]  # Giả sử mỗi vùng chỉ có 1 mã

        # Dữ liệu JSON
        json_result = extract_json_from_ocr(result)
        timestamp = datetime.now().strftime("%H:%M:%S")
        data = {timestamp: json_result[timestamp]}

        # Tạo đường dẫn tới file output
        file_path = f"{output_dir}/{stock_code}.json"

        # Đọc dữ liệu cũ nếu có
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # Gộp và ghi lại
        existing_data.update(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

    return prev_regions