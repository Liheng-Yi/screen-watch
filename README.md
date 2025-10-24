

## Prerequisites

Before running this program, you need to install:

1. **Python 3.8+**
2. **Tesseract OCR** (for card text recognition)
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

## Installation

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. If Tesseract is not in your PATH (Windows), edit `card_detector.py` and uncomment/modify this line:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```