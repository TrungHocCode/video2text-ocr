import json
from datetime import datetime
import cv2
import numpy as np
from ocr_process import split_frame_by_stock_count, classify_ocr_regions,extract_stock_codes_for_filenames
from collections import OrderedDict
def extract_json_from_ocr(ocr_result):
    trade_result = ocr_result['matched_trades']
    order_result = ocr_result['order_book']
    ref_time = datetime.now().strftime("%H:%M:%S")
    
    result_json = {
        ref_time: {
            "Khớp": {},
            "Giá": {},
            "KL": {},
            "M/B": {},
            "zone": []
        }
    }
    
    for idx, line in enumerate(trade_result):
        parts = line["text"].split()
        
        # Logic mới: flexible parsing
        if len(parts) >= 3:  # Tối thiểu cần: thời gian, giá, volume
            lan = f"Lần {idx + 1}"
            
            # Luôn lấy 3 phần tử đầu
            time = parts[0] if len(parts) > 0 else ""
            price = parts[1] if len(parts) > 1 else ""
            
            # Tìm volume và M/B
            volume = ""
            mb = "B"  # Mặc định là B
            
            if len(parts) >= 4:
                # Có đủ 4+ phần tử: time, price, diff(?), volume, mb(?)
                volume = parts[3] if len(parts) > 3 else parts[2]
                
                # Tìm M/B trong các phần tử cuối
                for part in parts[3:]:
                    if part.upper().startswith('M'):
                        mb = "M"
                        break
                    elif part.upper().startswith('B'):
                        mb = "B"
                        break
                    # Kiểm tra nếu part chứa M hoặc B ở cuối
                    elif len(part) > 0 and part[-1].upper() in ['M', 'B']:
                        mb = part[-1].upper()
                        break
                        
            elif len(parts) == 3:
                # Chỉ có 3 phần tử: time, price, volume/mb
                last_part = parts[2]
                
                # Kiểm tra xem phần tử cuối có phải là M/B không
                if last_part.upper() in ['M', 'B'] or (len(last_part) == 1 and last_part.upper() in ['M', 'B']):
                    mb = last_part.upper()
                    volume = ""  # Không có volume
                else:
                    volume = last_part
                    mb = "B"  # Mặc định
            
            # Gán vào JSON
            result_json[ref_time]["Khớp"][lan] = time
            result_json[ref_time]["Giá"][lan] = price
            result_json[ref_time]["KL"][lan] = volume
            result_json[ref_time]["M/B"][lan] = mb

    # Xử lý order book (giữ nguyên)
    for idx, order in enumerate(order_result):
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
def merged_json(order_json, trade_json):
    merged={}
    for time_key in set(order_json)|set(trade_json):
        merged[time_key]={
            'Khớp':trade_json.get(time_key, {}).get('Khớp', {}),
            'Giá': trade_json.get(time_key, {}).get('Giá', {}),
            'KL': trade_json.get(time_key, {}).get('KL', {}),
            'M/B': trade_json.get(time_key, {}).get('M/B', {}),
            'zone': order_json.get(time_key, {}).get('zone', [])
        }
    return merged
# def is_duplicate(dict1, dict2):
#     return dict1==dict2

def frame_changed(prev_region, curr_region, threshold=25, change_ratio=0.0005):
    if prev_region is None:
        return True
    diff = cv2.absdiff(prev_region, curr_region)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    
    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    ratio = changed_pixels / total_pixels
    return ratio > change_ratio
import matplotlib.pyplot as plt
def show_img(img):
    plt.imshow(img)
    plt.title("Ảnh đã crop")
    plt.axis('off')
    plt.show()
def save_json_per_stock(frame, ocr, prev_regions=None, stock_count=3, output_dir="./", stock_require=1, saved_bboxes=None,frame_id=1):
    stock_regions = split_frame_by_stock_count(frame, stock_count)

    if prev_regions is None:
        prev_regions = [None] * stock_count

    if saved_bboxes is None:
        saved_bboxes = [None] * stock_count

    for idx, region in enumerate(stock_regions):
        if idx == stock_require:
            break

        # Frame đầu tiên: OCR toàn bộ vùng để xác định bbox và stock_code
        if saved_bboxes[idx] is None:
            prev_regions[idx] = region.copy()

            ocr_result = ocr.ocr(region, cls=False)
            result = classify_ocr_regions(ocr_result)

            # Lưu bbox và stock_code
            stock_codes = extract_stock_codes_for_filenames(result['stocks'])
            if not stock_codes:
                continue

            stock_code = stock_codes[0]
            saved_bboxes[idx] = {
                "trade_bbox": result["trade_bbox"],
                "order_bbox": result["order_bbox"],
                "stock_code": stock_code
            }
            json_result = extract_json_from_ocr(result)
            timestamp = datetime.now().strftime("%H:%M:%S")
            data = {timestamp: json_result[timestamp]}
            # data = {frame_id: json_result[frame_id]}
            file_path = f"{output_dir}/{stock_code}.json"
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}

            existing_data.update(data)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)

        # Frame thứ hai trở đi: Chỉ OCR các vùng bbox đã lưu
        else:
            if not frame_changed(prev_regions[idx], region):
                continue

            prev_regions[idx] = region.copy()

            trade_bbox = saved_bboxes[idx]["trade_bbox"]
            order_bbox = saved_bboxes[idx]["order_bbox"]
            stock_code = saved_bboxes[idx]["stock_code"]
            if trade_bbox:
                x1, y1 = int(trade_bbox[0][0]), int(trade_bbox[0][1])
                x2, y2 = int(trade_bbox[2][0])+10, int(trade_bbox[2][1])
                trade_region = region[y1:y2, x1:x2]
                trade_ocr=ocr.ocr(trade_region,cls=False)
            if order_bbox:
                x1, y1 = int(order_bbox[0][0]), int(order_bbox[0][1])
                x2, y2 = int(order_bbox[2][0]), int(order_bbox[2][1])
                order_region = region[y1:y2, x1:x2]
                order_ocr=ocr.ocr(order_region,cls=False)
            order_result=classify_ocr_regions(order_ocr)
            trade_result=classify_ocr_regions(trade_ocr)  
            order_json=extract_json_from_ocr(order_result)
            trade_json=extract_json_from_ocr(trade_result)
            json_result = merged_json(order_json,trade_json)
            timestamp = datetime.now().strftime("%H:%M:%S")
            data = {timestamp: json_result[timestamp]}

            file_path = f"{output_dir}/{stock_code}.json"
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}

            existing_data.update(data)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)

    return prev_regions, saved_bboxes
def parse_time(time_str):
    return datetime.strptime(time_str, "%H:%M:%S")

def is_duplicate(dict1, dict2):
    return dict1 == dict2

def clean_ocr_json(data, group_size=5):
    # Sắp xếp theo timestamp tăng dần
    sorted_items = sorted(data.items(), key=lambda x: parse_time(x[0]))
    
    cleaned_data = OrderedDict()
    i = 0

    while i < len(sorted_items):
        # Lấy nhóm timestamps
        group = sorted_items[i:i+group_size]
        unique_group = []

        for j, (time_j, content_j) in enumerate(group):
            is_dup = False
            for (time_k, content_k) in unique_group:
                if is_duplicate(content_j, content_k):
                    is_dup = True
                    break
            if not is_dup:
                unique_group.append((time_j, content_j))

        # Thêm các bản ghi không trùng lặp vào kết quả
        for time_key, val in unique_group:
            cleaned_data[time_key] = val

        i += group_size

    return cleaned_data
