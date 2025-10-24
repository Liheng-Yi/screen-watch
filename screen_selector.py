"""GUI for selecting screen regions."""

import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, ImageTk, Image
import mss
import numpy as np
import cv2


class ScreenSelector:
    """Interactive screen region selector using tkinter."""
    
    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect = None
        self.selected_region = None
        self.sct = mss.mss()
    
    def get_monitors_info(self):
        """Get information about all connected monitors."""
        monitors = self.sct.monitors[1:]  # Skip the first one (all monitors combined)
        monitor_info = []
        
        for i, monitor in enumerate(monitors, 1):
            info = {
                'number': i,
                'left': monitor['left'],
                'top': monitor['top'],
                'width': monitor['width'],
                'height': monitor['height']
            }
            monitor_info.append(info)
        
        return monitor_info
    
    def choose_monitor(self):
        """Show a dialog to choose which monitor to use."""
        monitors = self.get_monitors_info()
        
        if len(monitors) == 1:
            print(f"Using Monitor 1: {monitors[0]['width']}x{monitors[0]['height']}")
            return monitors[0]
        
        # Create selection window
        root = tk.Tk()
        root.title("Select Monitor")
        root.geometry("500x400")
        root.configure(bg='#2b2b2b')
        
        selected_monitor = {'monitor': None}
        
        # Title
        title_label = tk.Label(
            root,
            text="Select Monitor for Poker Detection",
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Info label
        info_label = tk.Label(
            root,
            text=f"Detected {len(monitors)} monitor(s). Choose which one to monitor:",
            font=('Arial', 11),
            bg='#2b2b2b',
            fg='#cccccc'
        )
        info_label.pack(pady=10)
        
        # Frame for buttons
        button_frame = tk.Frame(root, bg='#2b2b2b')
        button_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        
        def select_monitor(monitor):
            selected_monitor['monitor'] = monitor
            root.quit()
            root.destroy()
        
        # Create button for each monitor with preview
        for monitor in monitors:
            frame = tk.Frame(button_frame, bg='#3b3b3b', relief=tk.RAISED, borderwidth=2)
            frame.pack(pady=10, padx=20, fill=tk.X)
            
            # Monitor info
            info_text = f"Monitor {monitor['number']}\n"
            info_text += f"Resolution: {monitor['width']} x {monitor['height']}\n"
            info_text += f"Position: ({monitor['left']}, {monitor['top']})"
            
            btn = tk.Button(
                frame,
                text=info_text,
                font=('Arial', 12),
                bg='#4a4a4a',
                fg='white',
                activebackground='#5a5a5a',
                activeforeground='white',
                command=lambda m=monitor: select_monitor(m),
                height=4,
                cursor='hand2'
            )
            btn.pack(pady=5, padx=5, fill=tk.X)
        
        root.mainloop()
        
        return selected_monitor['monitor']
        
    def select_region(self, monitor=None):
        """Launch fullscreen overlay to select a region.
        
        Args:
            monitor: Monitor info dict. If None, will prompt user to choose.
        """
        # Choose monitor if not specified
        if monitor is None:
            monitor = self.choose_monitor()
            if monitor is None:
                return None
        
        # Create fullscreen transparent window on specific monitor
        root = tk.Tk()
        
        # Position window on the selected monitor
        geometry = f"{monitor['width']}x{monitor['height']}+{monitor['left']}+{monitor['top']}"
        root.geometry(geometry)
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)
        root.attributes('-topmost', True)
        root.configure(bg='grey')
        
        canvas = tk.Canvas(root, cursor="cross", bg='grey')
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Store monitor offset for absolute coordinates
        monitor_offset_x = monitor['left']
        monitor_offset_y = monitor['top']
        
        def on_mouse_down(event):
            self.start_x = event.x
            self.start_y = event.y
            self.rect = canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline='red', width=3
            )
        
        def on_mouse_move(event):
            if self.rect:
                canvas.coords(
                    self.rect,
                    self.start_x, self.start_y, event.x, event.y
                )
        
        def on_mouse_up(event):
            self.end_x = event.x
            self.end_y = event.y
            
            # Calculate region (ensure positive width/height)
            # Convert to absolute screen coordinates
            x1 = min(self.start_x, self.end_x) + monitor_offset_x
            y1 = min(self.start_y, self.end_y) + monitor_offset_y
            x2 = max(self.start_x, self.end_x) + monitor_offset_x
            y2 = max(self.start_y, self.end_y) + monitor_offset_y
            
            self.selected_region = {
                'left': x1,
                'top': y1,
                'width': x2 - x1,
                'height': y2 - y1,
                'monitor': monitor['number']
            }
            
            root.quit()
            root.destroy()
        
        def on_escape(event):
            root.quit()
            root.destroy()
        
        canvas.bind('<ButtonPress-1>', on_mouse_down)
        canvas.bind('<B1-Motion>', on_mouse_move)
        canvas.bind('<ButtonRelease-1>', on_mouse_up)
        root.bind('<Escape>', on_escape)
        
        # Instructions
        label = tk.Label(
            root,
            text=f"Monitor {monitor['number']}: Click and drag to select the card region. Press ESC to cancel.",
            font=('Arial', 16),
            bg='grey',
            fg='white'
        )
        label.pack(pady=20)
        
        root.mainloop()
        
        return self.selected_region
    
    def select_point(self, title="Select Point", monitor=None):
        """Select a single point on screen (for fold button location).
        
        Args:
            title: Title text for the selection
            monitor: Monitor info dict. If None, will prompt user to choose.
        """
        # Choose monitor if not specified
        if monitor is None:
            monitor = self.choose_monitor()
            if monitor is None:
                return (None, None)
        
        # Create fullscreen transparent window on specific monitor
        root = tk.Tk()
        
        # Position window on the selected monitor
        geometry = f"{monitor['width']}x{monitor['height']}+{monitor['left']}+{monitor['top']}"
        root.geometry(geometry)
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)
        root.attributes('-topmost', True)
        root.configure(bg='grey')
        
        canvas = tk.Canvas(root, cursor="cross", bg='grey')
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Store monitor offset for absolute coordinates
        monitor_offset_x = monitor['left']
        monitor_offset_y = monitor['top']
        
        selected_point = {'x': None, 'y': None}
        
        def on_click(event):
            # Convert to absolute screen coordinates
            selected_point['x'] = event.x + monitor_offset_x
            selected_point['y'] = event.y + monitor_offset_y
            root.quit()
            root.destroy()
        
        def on_escape(event):
            root.quit()
            root.destroy()
        
        canvas.bind('<ButtonPress-1>', on_click)
        root.bind('<Escape>', on_escape)
        
        # Instructions
        label = tk.Label(
            root,
            text=f"Monitor {monitor['number']} - {title}: Click on the location. Press ESC to cancel.",
            font=('Arial', 16),
            bg='grey',
            fg='white'
        )
        label.pack(pady=20)
        
        root.mainloop()
        
        return (selected_point['x'], selected_point['y'])


def preview_selection(region):
    """Show a preview of the selected region."""
    import cv2
    
    with mss.mss() as sct:
        monitor = {
            "left": region['left'],
            "top": region['top'],
            "width": region['width'],
            "height": region['height']
        }
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize if too large
        max_height = 600
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            width = int(img.shape[1] * scale)
            img = cv2.resize(img, (width, max_height))
        
        cv2.imshow('Selected Region Preview (Press any key to close)', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test the selector
    selector = ScreenSelector()
    
    # Show available monitors
    monitors = selector.get_monitors_info()
    print(f"\nDetected {len(monitors)} monitor(s):")
    for m in monitors:
        print(f"  Monitor {m['number']}: {m['width']}x{m['height']} at ({m['left']}, {m['top']})")
    
    print("\nSelect the card region...")
    region = selector.select_region()
    
    if region:
        print(f"\nSelected region: {region}")
        preview_selection(region)

