"""Main application for poker screen monitoring."""

import cv2
import time
import json
import os
from typing import Optional, Dict, List, Tuple
import pyautogui
import numpy as np
import winsound

from screen_selector import ScreenSelector
from screen_capture import ScreenCapture
from card_detector import CardDetector
import config


class PokerMonitor:
    """Main poker monitoring application."""
    
    def __init__(self):
        """Initialize the poker monitor."""
        self.selector = ScreenSelector()
        self.detector = CardDetector()
        
        self.card1_region = None
        self.card2_region = None
        self.capture1 = None
        self.capture2 = None
        self.fold_button_position = None
        self.is_running = False
        
        # Load saved settings if they exist
        self.load_settings()
        
    def load_settings(self):
        """Load saved settings from file."""
        if os.path.exists('settings.json'):
            try:
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                    self.card1_region = settings.get('card1_region')
                    self.card2_region = settings.get('card2_region')
                    self.fold_button_position = settings.get('fold_button_position')
                    
                    # Initialize capture objects if regions exist
                    if self.card1_region:
                        self.capture1 = ScreenCapture()
                        self.capture1.set_region(self.card1_region)
                    if self.card2_region:
                        self.capture2 = ScreenCapture()
                        self.capture2.set_region(self.card2_region)
                    
                    print("Loaded saved settings")
            except Exception as e:
                print(f"Error loading settings: {e}")
    
    def save_settings(self):
        """Save current settings to file."""
        settings = {
            'card1_region': self.card1_region,
            'card2_region': self.card2_region,
            'fold_button_position': self.fold_button_position
        }
        try:
            with open('settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            print("Settings saved")
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def setup_card_regions(self):
        """Let user select the two card regions."""
        print("\n=== Select Card Regions ===")
        
        # Show available monitors
        monitors = self.selector.get_monitors_info()
        print(f"\nDetected {len(monitors)} monitor(s):")
        for m in monitors:
            print(f"  Monitor {m['number']}: {m['width']}x{m['height']} at position ({m['left']}, {m['top']})")
        
        # Select first card region
        print("\n--- FIRST CARD ---")
        print("Click and drag to select the area where your FIRST card appears")
        print("(Select just one card, as small as possible around the card rank)")
        region1 = self.selector.select_region()
        
        if not region1 or region1['width'] <= 0 or region1['height'] <= 0:
            print("Invalid region selected for first card")
            return False
        
        self.card1_region = region1
        self.capture1 = ScreenCapture()
        self.capture1.set_region(region1)
        print(f"✓ First card region set: {region1}")
        
        # Select second card region
        print("\n--- SECOND CARD ---")
        print("Click and drag to select the area where your SECOND card appears")
        print("(Select just one card, as small as possible around the card rank)")
        region2 = self.selector.select_region()
        
        if not region2 or region2['width'] <= 0 or region2['height'] <= 0:
            print("Invalid region selected for second card")
            return False
        
        self.card2_region = region2
        self.capture2 = ScreenCapture()
        self.capture2.set_region(region2)
        print(f"✓ Second card region set: {region2}")
        
        self.save_settings()
        return True
    
    def setup_fold_button(self):
        """Let user select the fold button location."""
        print("\n=== Select Fold Button ===")
        
        # Show available monitors
        monitors = self.selector.get_monitors_info()
        print(f"\nDetected {len(monitors)} monitor(s):")
        for m in monitors:
            print(f"  Monitor {m['number']}: {m['width']}x{m['height']} at position ({m['left']}, {m['top']})")
        
        print("\nYou will be prompted to choose a monitor, then click on the fold button.")
        position = self.selector.select_point("Fold Button Location")
        
        if position[0] is not None and position[1] is not None:
            self.fold_button_position = position
            print(f"Fold button position set: {position}")
            self.save_settings()
            return True
        else:
            print("Invalid position selected")
            return False
    
    def test_detection(self):
        """Test card detection on current screen."""
        if not self.card1_region or not self.card2_region:
            print("Please set up card regions first")
            return
        
        print("\n=== Testing Card Detection ===")
        
        # Capture both cards
        img1 = self.capture1.capture()
        img2 = self.capture2.capture()
        
        if img1 is not None and img2 is not None:
            print("Detecting cards...")
            rank1 = self.detector.detect_single_card(img1)
            rank2 = self.detector.detect_single_card(img2)
            
            print(f"Card 1: {rank1 if rank1 else 'Not detected'}")
            print(f"Card 2: {rank2 if rank2 else 'Not detected'}")
            
            # Show visualization
            cv2.imshow('Card 1 (Press any key to continue)', img1)
            cv2.waitKey(0)
            cv2.imshow('Card 2 (Press any key to close)', img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to capture screen")
    
    def click_fold_button(self):
        """Click the fold button without moving mouse cursor."""
        if not self.fold_button_position:
            return False
        
        try:
            x, y = self.fold_button_position
            # Get current mouse position
            current_x, current_y = pyautogui.position()
            # Click at fold button position
            pyautogui.click(x, y)
            # Immediately move back to original position
            pyautogui.moveTo(current_x, current_y, duration=0)
            return True
        except Exception as e:
            print(f" [Error clicking: {e}]")
            return False
    
    def should_fold(self, rank1: Optional[str], rank2: Optional[str]) -> Tuple[bool, str]:
        """
        Determine if the current hand should fold.
        
        Fold unless:
        - Pocket pair (any pair)
        - Has an Ace
        - Has a King
        - Both cards are high cards (A, K, Q, J)
        - 10-J combination
        - Connectors (consecutive ranks)
        
        Args:
            rank1: Rank of first card
            rank2: Rank of second card
            
        Returns:
            Tuple of (should_fold, decision_reason)
        """
        if not rank1 or not rank2:
            return False, ""
        
        # Define high cards
        high_cards = {'A', 'K', 'Q', 'J'}
        
        # Rank values for connector detection
        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
            '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        
        # Don't fold if it's a pocket pair
        if rank1 == rank2:
            return False, f"Pair"
        
        # Don't fold if has an Ace
        if rank1 == 'A' or rank2 == 'A':
            return False, f"Ace"
        
        # Don't fold if has a King (but exclude weak Kings: K2-K9)
        if rank1 == 'K' or rank2 == 'K':
            other_card = rank2 if rank1 == 'K' else rank1
            weak_kickers = {'2', '3', '4'}
            if other_card not in weak_kickers:
                return False, f"King"
        
        # Don't fold if both cards are high cards (A, K, Q, J)
        if rank1 in high_cards and rank2 in high_cards:
            return False, f"High"
        
        # Don't fold if 10-J combination
        if (rank1 == '10' and rank2 == 'J') or (rank1 == 'J' and rank2 == '10'):
            return False, f"10-J"
        
        # Don't fold if connectors (consecutive ranks)
        val1 = rank_values.get(rank1, 0)
        val2 = rank_values.get(rank2, 0)
        if abs(val1 - val2) == 1:
            # Exclude weak connectors: 2-3, 3-4, 4-5
            min_val = min(val1, val2)
            if min_val >= 5:  # Only play connectors starting from 5-6 and up
                return False, f"Connector"
        
        # Special case: A-2 is also a connector (but it's weak, so we fold it)
        # Removed A-2 connector logic to fold it
        
        # Fold everything else
        return True, f"FOLD"
    
    def run_monitoring(self, show_preview=True):
        """Run the main monitoring loop.
        
        Args:
            show_preview: If True, shows live preview window
        """
        if not self.card1_region or not self.card2_region:
            print("Please set up card regions first")
            return
        
        print("\n=== Starting Monitoring ===")
        print("Press Ctrl+C in console or 'Q' in preview window to stop")
        print(f"Monitoring at {config.CAPTURE_FPS} FPS")
        print(f"Show preview: {show_preview}")
        print("-" * 50)
        
        self.is_running = True
        frame_time = 1.0 / config.CAPTURE_FPS
        last_hand = (None, None)
        frame_count = 0
        
        try:
            while self.is_running:
                start_time = time.time()
                frame_count += 1
                
                # Capture both cards
                img1 = self.capture1.capture()
                img2 = self.capture2.capture()
                
                if img1 is not None and img2 is not None:
                    # Detect both cards
                    rank1 = self.detector.detect_single_card(img1)
                    rank2 = self.detector.detect_single_card(img2)
                    
                    current_hand = (rank1, rank2)
                    
                    # Only process if cards changed
                    if current_hand != last_hand:
                        timestamp = time.strftime("%H:%M:%S")
                        if rank1 and rank2:  # Both cards detected
                            # Check if should fold
                            should_fold, reason = self.should_fold(rank1, rank2)
                            
                            # Print hand history
                            print(f"[{timestamp}] {rank1}|{rank2} → {reason}")
                            
                            if should_fold and self.fold_button_position:
                                self.click_fold_button()
                                time.sleep(config.FOLD_DELAY)
                            else:
                                # Play notification sound for playable hands (no console output)
                                try:
                                    winsound.Beep(1000, 200)  # 1000 Hz for 200ms
                                except:
                                    pass
                        elif rank1 or rank2:  # Only one card detected
                            print(f"[{timestamp}] {rank1 if rank1 else '?'}|{rank2 if rank2 else '?'}")
                        else:
                            print(f"[{timestamp}] No cards")
                        
                        last_hand = current_hand
                    
                    # Show live preview
                    if show_preview:
                        # Create combined visualization
                        h1, w1 = img1.shape[:2]
                        h2, w2 = img2.shape[:2]
                        
                        # Scale up images for visibility (5x smaller than before)
                        scale_factor = 5
                        img1_scaled = cv2.resize(img1, (w1 * scale_factor, h1 * scale_factor), 
                                                interpolation=cv2.INTER_NEAREST)
                        img2_scaled = cv2.resize(img2, (w2 * scale_factor, h2 * scale_factor), 
                                                interpolation=cv2.INTER_NEAREST)
                        
                        # Make both images same height for side-by-side display
                        h1_s, w1_s = img1_scaled.shape[:2]
                        h2_s, w2_s = img2_scaled.shape[:2]
                        max_height = max(h1_s, h2_s)
                        
                        if h1_s < max_height:
                            img1_scaled = cv2.resize(img1_scaled, (int(w1_s * max_height / h1_s), max_height))
                        if h2_s < max_height:
                            img2_scaled = cv2.resize(img2_scaled, (int(w2_s * max_height / h2_s), max_height))
                        
                        # Add some padding between cards
                        padding = np.ones((max_height, 10, 3), dtype=np.uint8) * 50
                        
                        # Combine images side by side with padding
                        display_img = np.hstack([img1_scaled, padding, img2_scaled])
                        
                        # Add overlay with info (smaller)
                        overlay = display_img.copy()
                        cv2.rectangle(overlay, (0, 0), (display_img.shape[1], 50), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)
                        
                        # Add text (smaller font)
                        cards_text = f"{rank1 if rank1 else '?'} | {rank2 if rank2 else '?'}"
                        color = (0, 255, 0) if (rank1 and rank2) else (0, 165, 255)
                        cv2.putText(display_img, cards_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Show window with always on top flag
                        window_name = 'Poker Monitor - Press Q to quit'
                        cv2.imshow(window_name, display_img)
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                        
                        # Check for quit key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                            print("\nStopping monitoring...")
                            break
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
        finally:
            self.is_running = False
            if show_preview:
                cv2.destroyAllWindows()
    
    def show_menu(self):
        """Show interactive menu."""
        while True:
            print("\n" + "="*50)
            print("POKER SCREEN MONITOR")
            print("="*50)
            print("1. Set up card regions and start monitoring")
            print("2. Set up fold button location")
            print("3. Test card detection")
            print("4. Exit")
            print("="*50)
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                if self.setup_card_regions():
                    # Check if fold button is configured
                    if not self.fold_button_position:
                        print("\n⚠️  Fold button not configured yet.")
                        self.setup_fold_button()
                    
                    # Automatically start monitoring after setup
                    print("\n✓ Setup complete! Starting monitoring in 2 seconds...")
                    time.sleep(2)
                    self.run_monitoring(show_preview=True)
            elif choice == '2':
                self.setup_fold_button()
            elif choice == '3':
                self.test_detection()
            elif choice == '4':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")


def main():
    """Main entry point."""
    print("Poker Screen Monitor")
    print("=" * 50)
    
    monitor = PokerMonitor()
    monitor.show_menu()


if __name__ == "__main__":
    main()


