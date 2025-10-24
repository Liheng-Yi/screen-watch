"""Card detection using OCR and image processing."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import pytesseract
import os


class CardDetector:
    """Detects playing cards from screen captures."""
    
    def __init__(self):
        """Initialize card detector."""
        # Set tesseract path if needed (Windows)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        self.rank_map = {
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
            '7': '7', '8': '8', '9': '9', '10': '10',
            'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
            # Common OCR misreads
            'T': '10', 'I': '1', 'O': '0', 'l': '1', '|': '1'
        }
        
        self.valid_ranks = {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}
        
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better card detection.
        
        Args:
            img: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def find_card_regions(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find potential card regions in the image.
        
        Args:
            img: Input image
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        processed = self.preprocess_image(img)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area (adjust these values based on your poker client)
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                # Cards typically have certain aspect ratios
                aspect_ratio = w / h if h > 0 else 0
                if 0.4 < aspect_ratio < 1.5:
                    card_regions.append((x, y, w, h))
        
        return card_regions
    
    def extract_text_from_region(self, img: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """
        Extract text from a specific region using OCR.
        
        Args:
            img: Input image
            region: Bounding box (x, y, w, h)
            
        Returns:
            Detected text
        """
        x, y, w, h = region
        roi = img[y:y+h, x:x+w]
        
        # Preprocess
        processed = self.preprocess_image(roi)
        
        # Upscale for better OCR
        scale = 3
        processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # OCR configuration for single characters
        custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=23456789TJQKA'
        text = pytesseract.image_to_string(processed, config=custom_config).strip()
        
        return text
    
    def detect_cards_simple(self, img: np.ndarray) -> List[Tuple[str, str]]:
        """
        Simple card detection using OCR on the entire image.
        More reliable for poker clients with clear card display.
        
        Args:
            img: Input image
            
        Returns:
            List of detected cards as (rank, suit) tuples
        """
        detected_cards = []
        
        # Preprocess the entire image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple thresholding methods
        methods = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ]
        
        all_detected = set()
        
        for processed in methods:
            # Upscale for better OCR
            scale = 2
            scaled = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # OCR configuration for card ranks
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=23456789TJQKA'
            text = pytesseract.image_to_string(scaled, config=custom_config).strip()
            
            # Parse detected text
            for char in text:
                if char in self.rank_map:
                    rank = self.rank_map[char]
                    all_detected.add(rank)
        
        # Convert to list of tuples
        for rank in sorted(all_detected):
            detected_cards.append((rank, 'unknown'))
        
        return detected_cards
    
    def detect_single_card(self, img: np.ndarray) -> Optional[str]:
        """
        Detect a single card rank from an image.
        Optimized for detecting one card with high accuracy.
        
        Args:
            img: Input image containing a single card
            
        Returns:
            Detected card rank or None
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple preprocessing methods for better "10" detection
        all_ranks = []
        
        # Method 1: Otsu thresholding (usually best)
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        all_ranks.extend(self._extract_ranks_from_image(binary1, scale=5))
        
        # Method 2: Inverted Otsu (for dark text on light background)
        binary1_inv = cv2.bitwise_not(binary1)
        all_ranks.extend(self._extract_ranks_from_image(binary1_inv, scale=5))
        
        # Method 3: High threshold
        _, binary2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        all_ranks.extend(self._extract_ranks_from_image(binary2, scale=5))
        
        # Method 4: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        all_ranks.extend(self._extract_ranks_from_image(adaptive, scale=5))
        
        # Count occurrences and return most common valid rank
        if all_ranks:
            from collections import Counter
            rank_counts = Counter(all_ranks)
            # Get the most common rank
            most_common = rank_counts.most_common(1)[0][0]
            return most_common
        
        return None
    
    def _extract_ranks_from_image(self, img: np.ndarray, scale: int = 4) -> List[str]:
        """
        Extract card ranks from a preprocessed image using OCR.
        
        Args:
            img: Preprocessed binary image
            scale: Upscaling factor for better OCR
            
        Returns:
            List of detected valid ranks
        """
        detected_ranks = []
        
        try:
            # Upscale for better OCR
            scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Apply morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            scaled = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)
            
            # Try multiple PSM modes - more comprehensive for "10"
            psm_modes = [
                6,  # Uniform block of text
                7,  # Single line
                8,  # Single word
            ]
            
            for psm in psm_modes:
                try:
                    # More permissive config for "10"
                    config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=A023456789TJQK10'
                    text = pytesseract.image_to_string(scaled, config=config).strip()
                    text_clean = text.upper().replace(' ', '').replace('\n', '').replace('\t', '')
                    
                    # Check for "10" first (two characters together)
                    if '10' in text_clean:
                        detected_ranks.append('10')
                        continue  # If we found 10, skip individual parsing
                    
                    # Check for 'T' or 'I' followed by 'O' or '0' (common misreads of "10")
                    if len(text_clean) >= 2:
                        for i in range(len(text_clean) - 1):
                            if text_clean[i] in ['1', 'I', 'T', 'l', '|'] and text_clean[i+1] in ['0', 'O']:
                                detected_ranks.append('10')
                                break
                    
                    # Parse individual characters
                    i = 0
                    while i < len(text_clean):
                        # Check for "10" as a unit
                        if i < len(text_clean) - 1 and text_clean[i:i+2] == '10':
                            detected_ranks.append('10')
                            i += 2
                        elif text_clean[i] in self.rank_map:
                            rank = self.rank_map[text_clean[i]]
                            if rank in self.valid_ranks:
                                detected_ranks.append(rank)
                            i += 1
                        else:
                            i += 1
                except Exception:
                    pass
        except Exception:
            pass
        
        return detected_ranks
    
    def detect_cards(self, img: np.ndarray) -> List[Tuple[str, str]]:
        """
        Detect cards from an image.
        
        Args:
            img: Input image
            
        Returns:
            List of detected cards as (rank, suit) tuples
        """
        # Try simple detection first (works better for most poker clients)
        simple_cards = self.detect_cards_simple(img)
        
        if simple_cards:
            return simple_cards
        
        # Fallback to region-based detection
        detected_cards = []
        regions = self.find_card_regions(img)
        
        for region in regions:
            text = self.extract_text_from_region(img, region)
            
            # Try to parse rank
            for char in text:
                if char in self.rank_map:
                    rank = self.rank_map[char]
                    detected_cards.append((rank, 'unknown'))
                    break
        
        return detected_cards
    
    def visualize_detection(self, img: np.ndarray, cards: List[Tuple[str, str]]) -> np.ndarray:
        """
        Draw detected cards on image for visualization.
        
        Args:
            img: Input image
            cards: Detected cards
            
        Returns:
            Image with annotations
        """
        result = img.copy()
        
        # Draw text with detected cards
        y_offset = 30
        for i, (rank, suit) in enumerate(cards):
            text = f"Card {i+1}: {rank} of {suit}"
            cv2.putText(result, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        return result


if __name__ == "__main__":
    # Test card detection
    from screen_selector import ScreenSelector
    from screen_capture import ScreenCapture
    
    selector = ScreenSelector()
    print("Select the region containing your cards...")
    region = selector.select_region()
    
    if region:
        capture = ScreenCapture(region)
        detector = CardDetector()
        
        img = capture.capture()
        if img is not None:
            print("Detecting cards...")
            cards = detector.detect_cards(img)
            print(f"Detected cards: {cards}")
            
            # Visualize
            result = detector.visualize_detection(img, cards)
            cv2.imshow('Card Detection (Press any key to close)', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

