# Poker Screen Monitor

An automated poker hand monitoring system that uses OCR to detect cards and make decisions based on configurable strategies.

## Features

- **Dual OCR Engine**: Uses EasyOCR (deep learning-based) as primary method with Tesseract fallback
- **High Accuracy**: Multi-method preprocessing and voting system for reliable card detection
- **Automated Folding**: Automatically clicks fold button for weak hands
- **Visual Feedback**: Live preview window showing detected cards
- **Sound Notifications**: Beep alerts for playable hands
- **Persistent Settings**: Saves card regions and button positions
- **Combined Capture**: Select one region containing both card ranks for faster setup
- **Flexible Strategy**: Configurable hand evaluation rules

## Prerequisites

Before running this program, you need to install:

1. **Python 3.8+**
2. **Tesseract OCR** (for fallback card text recognition)
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

## Installation

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

**Note**: EasyOCR will download language models (~100MB) on first run. This is normal and only happens once.

3. If Tesseract is not in your PATH (Windows), edit `card_detector.py` and modify this line:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## OCR Engines

### EasyOCR (Primary - Recommended)
- **Accuracy**: Higher accuracy using deep learning models
- **Speed**: Slightly slower but more reliable
- **Setup**: Automatically downloads on first run
- **GPU**: Can use GPU acceleration if available (set `gpu=True` in code)

### Tesseract (Fallback)
- **Accuracy**: Good for clear, high-contrast text
- **Speed**: Very fast
- **Setup**: Requires separate installation
- **Use Case**: Automatic fallback if EasyOCR fails or isn't installed

The system automatically uses EasyOCR if available, falling back to Tesseract when needed.

## Selecting Card Regions

When you choose option `1` in the menu, you'll be prompted to select a single region that contains both card ranks. The detector automatically finds the two ranks in that area. You can rerun the setup at any time to recapture the region. All selections are saved to `settings.json`.