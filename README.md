# video2text-ocr

**video2text-ocr** is a modular pipeline for extracting and processing Vietnamese stock trading data from video files using Optical Character Recognition (OCR). It performs four main tasks:

1. Extracts frames from input video at predefined intervals
2. Applies OCR to detect and extract text from each frame
3. Classifies and processes trading data and order book information
4. Saves results in JSON format organized by stock codes

## Features

- **Multi-format Video Support**: Works with various video formats (MP4, AVI, MOV, etc.)
- **Configurable Frame Extraction**: Extract frames at custom time intervals
- **Vietnamese OCR Specialization**: Uses PaddleOCR for accurate Vietnamese text recognition
- **Change Detection**: Automatically identifies and processes only changed frames
- **Smart Detecting**: Automatically detects stocks count and bounding box of content
- **Smart Region Splitting**: Automatically divides frames into regions based on stock count
- **JSON Data Export**: Saves results in structured JSON format
- **Multi-stock Processing**: Processes multiple stock codes simultaneously within frames
- **Comprehensive Logging**: Detailed progress tracking with full logging system
## Limitations
- **Fixed Region Splitting**: This system only supports videos that are divided into three equally sized horizontal regions and 2:2 regions. Videos with a different number of stocks or regions of unequal widths are not currently supported. (This limitation will be addressed in future updates to support more flexible and customizable frame splitting.)
## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- PaddleOCR

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/video2text-ocr.git
cd video2text-ocr

# Install Python dependencies
pip install -r requirements.txt
```
## Project Structure

```
video2text-ocr/
├── src/
├├── main.py              # Main execution file
├├── get_frames.py        # Frame extraction module
├├── ocr_process.py       # OCR processing and data classification module
├├── json_process.py      # JSON processing and saving module
├├── logger_config.py     # Logging configuration
├── requirements.txt      # Dependencies list            
├── output/               # Directory for JSON results
└├── app.log              # Log file
```

## Usage
**Important**: Before running the project, you must create an output folder to save your results, or you can change the directory path in main.py.
# Create output directory first
mkdir output

# Then run the program
python main.py
### Basic Usage
Before run this project, you must create a folder output to save your result or you can change directory in file main.py t
```bash
python src/main.py
```

The program will prompt you to enter:
1. Path to the video file
2. Number of stock codes to process

### Example

```bash
python src/main.py
# Enter video path: ./stock_video.mp4
```
