import json
import cv2
import numpy as np
from ocr_process import split_frame_by_stock_count, classify_ocr_regions,extract_stock_codes_for_filenames
from datetime import timedelta
def extract_json_from_ocr(ocr_result,video_time):
    trade_result = ocr_result['matched_trades']
    order_result = ocr_result['order_book']
    result_json = {
        video_time: {
            "Khớp": {},
            "Giá": {},
            "KL": {},
            "M/B": {},
            "zone": []
        }
    }
    for idx, line in enumerate(trade_result):
        parts = line["text"].split()

        if len(parts) >= 3:
            lan = f"Lần {idx + 1}"

            time = parts[0] if len(parts) > 0 else ""
            price = parts[1] if len(parts) > 1 else ""

            volume = ""
            mb = "B"
            
            if len(parts) >= 4:
                volume = parts[3] if len(parts) > 3 else parts[2]
                for part in parts[3:]:
                    if part.upper().startswith('M'):
                        mb = "M"
                        break
                    elif part.upper().startswith('B'):
                        mb = "B"
                        break
                    elif len(part) > 0 and part[-1].upper() in ['M', 'B']:
                        mb = part[-1].upper()
                        break
                        
            elif len(parts) == 3:
                last_part = parts[2]
                if last_part.upper() in ['M', 'B'] or (len(last_part) == 1 and last_part.upper() in ['M', 'B']):
                    mb = last_part.upper()
                    volume = ""  
                else:
                    volume = last_part
                    mb = "B"  
            result_json[video_time]["Khớp"][lan] = time
            result_json[video_time]["Giá"][lan] = price
            result_json[video_time]["KL"][lan] = volume
            result_json[video_time]["M/B"][lan] = mb

    for idx, order in enumerate(order_result):
        parts = order["text"].split()
        if len(parts) >= 4:
            zone = {
                f"KL_mua {idx + 1}": parts[0],
                f"Gia_mua {idx + 1}": parts[1],
                f"Gia_ban {idx + 1}": parts[2],
                f"KL_ban {idx + 1}": parts[3]
            }
            result_json[video_time]["zone"].append(zone)

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
def format_seconds_to_time(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

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
def extract_first_frame(ocr,frame):
    """
    Phân tích frame đầu tiên để lấy số lượng mã, và thông tin bbox (trade + order) cho từng mã.
    """
    # Tách các vùng theo số mã cổ phiếu
    first_result=classify_ocr_regions(ocr.ocr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),cls=False))['stocks']
    stocks_count=int(len(first_result))
    stock_codes=first_result
    stock_regions=split_frame_by_stock_count(frame,stocks_count)
    stock_infos = []
    for idx, region in enumerate(stock_regions):
        ocr_result=ocr.ocr(region,cls=False)
        result=classify_ocr_regions(ocr_result)

        stock_code = stock_codes[idx]['text']
        trade_bbox = result.get("trade_bbox", [])
        order_bbox = result.get("order_bbox", [])
        
        stock_infos.append({
            "stock_code": stock_code,
            "trade_bbox": trade_bbox,
            "order_bbox": order_bbox
        })

    return {
        "stock_count": len(stock_infos),
        "stocks": stock_infos
    }
def save_json_per_stock(frame, ocr, prev_regions=None, stock_count=3, output_dir="./", saved_bboxes=None,frame_id=1):
    stock_regions = split_frame_by_stock_count(frame, stock_count)

    if prev_regions is None:
        prev_regions = [None] * stock_count
    for idx, region in enumerate(stock_regions):
        if frame_id!=1:
            if not frame_changed(prev_regions[idx], region):
                continue
        prev_regions[idx] = region.copy()
        trade_bbox = saved_bboxes[idx]["trade_bbox"]
        order_bbox = saved_bboxes[idx]["order_bbox"]
        stock_code = saved_bboxes[idx]["stock_code"]
        if trade_bbox:
            x1, y1 = int(trade_bbox[0][0]), int(trade_bbox[0][1])
            x2, y2 = int(trade_bbox[2][0])+20, int(trade_bbox[2][1]+20)
            trade_region = region[y1:y2, x1:x2]
            trade_ocr=ocr.ocr(trade_region,cls=False)
        if order_bbox:
            x1, y1 = int(order_bbox[0][0]), int(order_bbox[0][1])
            x2, y2 = int(order_bbox[2][0]), int(order_bbox[2][1])
            order_region = region[y1:y2, x1:x2]
            order_ocr=ocr.ocr(order_region,cls=False)
        video_time=round(frame_id*0.4,2)

        order_result=classify_ocr_regions(order_ocr)
        trade_result=classify_ocr_regions(trade_ocr)

        order_json=extract_json_from_ocr(order_result,video_time)
        trade_json=extract_json_from_ocr(trade_result,video_time)

        json_result = merged_json(order_json,trade_json)
        data = {video_time: json_result[video_time]}
        file_path = f"{output_dir}/{stock_code}.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}
        existing_data.update(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    return prev_regions

