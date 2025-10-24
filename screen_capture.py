"""Screen capture functionality."""

import mss
import numpy as np
import cv2
from typing import Optional, Dict


class ScreenCapture:
    """Handles screen capture operations."""
    
    def __init__(self, region: Optional[Dict[str, int]] = None):
        """
        Initialize screen capture.
        
        Args:
            region: Dict with 'left', 'top', 'width', 'height' keys
        """
        self.sct = mss.mss()
        self.region = region
        
    def set_region(self, region: Dict[str, int]):
        """Set the capture region."""
        self.region = region
        
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture the screen region.
        
        Returns:
            numpy array of the captured image (BGR format) or None if no region set
        """
        if not self.region:
            return None
            
        monitor = {
            "left": self.region['left'],
            "top": self.region['top'],
            "width": self.region['width'],
            "height": self.region['height']
        }
        
        screenshot = self.sct.grab(monitor)
        # Convert to numpy array and BGR format (for OpenCV)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def capture_and_save(self, filename: str = "capture.png") -> bool:
        """
        Capture and save to file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        img = self.capture()
        if img is not None:
            cv2.imwrite(filename, img)
            return True
        return False
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'sct'):
            self.sct.close()


if __name__ == "__main__":
    # Test screen capture
    from screen_selector import ScreenSelector
    
    selector = ScreenSelector()
    print("Select a region to capture...")
    region = selector.select_region()
    
    if region:
        capture = ScreenCapture(region)
        img = capture.capture()
        
        if img is not None:
            cv2.imshow('Captured Image (Press any key to close)', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            capture.capture_and_save("test_capture.png")
            print("Saved capture to test_capture.png")

