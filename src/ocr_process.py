import re
from unidecode import unidecode

def normalize_text(text):
    text = unidecode(text)
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w]", "_", text)
    return text

def clean_stock_code(raw_code):
    raw_code = re.sub(r'[^A-Z]', '', raw_code.upper())  # Chỉ giữ chữ in hoa
    if len(raw_code) > 3:
        return raw_code[-3:]  # Lấy 7 ký tự cuối
    return raw_code

def split_frame_by_stock_count(frame, stock_count, axis='vertical'):
    h, w = frame.shape[:2]
    regions = []
    if axis == 'vertical':
        region_width = w // stock_count
        for i in range(stock_count):
            x1 = i * region_width
            x2 = (i + 1) * region_width if i < stock_count - 1 else w
            region = frame[:, x1:x2]
            regions.append(region)
    else:
        region_height = h // stock_count
        for i in range(stock_count):
            y1 = i * region_height
            y2 = (i + 1) * region_height if i < stock_count - 1 else h
            region = frame[y1:y2, :]
            regions.append(region)
    return regions
def group_entries_by_line(ocr_entries, y_threshold=10):
    lines = []
    for entry in sorted(ocr_entries, key=lambda x: x[0][0][1]):  # sort by y
        added = False
        for line in lines:
            if abs(entry[0][0][1] - line[0][0][0][1]) <= y_threshold:
                line.append(entry)
                added = True
                break
        if not added:
            lines.append([entry])
    return lines
def compute_union_bbox(bboxes):
    all_x = [pt[0] for bbox in bboxes for pt in bbox]
    all_y = [pt[1] for bbox in bboxes for pt in bbox]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

def detect_trade_and_order_lines(grouped_lines, max_trade_lines=2):
    import re
    regex_trade = re.compile(r"\d{2}:\d{2}:\d{2}.*M")
    regex_order = re.compile(r"\d{1,3}(?:,\d{3})?\s+\d{2}\.\d{2}\s+\d{2}\.\d{2}\s+\d{1,3}(?:,\d{3})?")

    trade_entries = []
    order_entries = []

    for line in grouped_lines:
        sorted_line = sorted(line, key=lambda x: x[0][0][0])  # sort by x
        text_line = " ".join([e[1][0] for e in sorted_line])
        bbox = compute_union_bbox([e[0] for e in sorted_line])

        if regex_trade.match(text_line):
            trade_entries.append({'text': text_line, 'bbox': bbox})
        elif regex_order.match(text_line):
            order_entries.append({'text': text_line, 'bbox': bbox})

    # Nếu có ít nhất 2 dòng trade, gộp bbox của 2 dòng đầu và gán lại cho cả 2
    if len(trade_entries) >= 2:
        bbox_union = compute_union_bbox([trade_entries[0]['bbox'], trade_entries[1]['bbox']])
        trade_entries[0]['bbox'] = bbox_union
        trade_entries[1]['bbox'] = bbox_union
        trade_entries = trade_entries[:2]  # chỉ lấy 2 dòng đầu

    return trade_entries, order_entries
def classify_ocr_regions(result):
    entries = result[0]

    # Bước 1: nhóm theo dòng
    grouped_lines = group_entries_by_line(entries)

    # Bước 2: tách từng loại dòng
    trade_lines, order_lines = detect_trade_and_order_lines(grouped_lines)

    # Bước 3: tìm vùng stock name
    stock_names = []
    for entry in entries:
        text = entry[1][0]
        if any(k in text for k in ("CTCP", "TMCP", "Chỉ số")):
            stock_names.append({
                "text": normalize_text(text),
                "coords": (
                    int(entry[0][0][0]), int(entry[0][0][1]),
                    int(entry[0][2][0]), int(entry[0][2][1])
                )
            })
    # num_stocks=len(stock_names)/2
    # print(f"Đã nhận diện được {num_stocks} mã chứng khoán")
    return {
        "stocks": stock_names,
        "order_book": order_lines,
        "matched_trades": trade_lines
    }

def extract_stock_codes_for_filenames(stocks):
    codes = []
    for stock in stocks:
        # Làm sạch chuỗi OCR
        text = unidecode(stock["text"])
        text = re.sub(r"[_\s]+", " ", text).strip()
        parts = text.split()

        # Tìm phần có dạng viết hoa (tối đa 5 ký tự), không chứa CTCP
        code = next((p for p in parts if p.isupper() and 2 <= len(p) <= 5 and "CTCP" not in p), None)
        if code:
            clean_code = re.sub(r"[^\w]", "", code)  # loại bỏ ký tự không hợp lệ trong tên file
            codes.append(clean_code)
    
    return codes
