"""
Screen snipping tool for capturing a region of the screen.
Creates a fullscreen transparent overlay where the user can click and drag
to select a rectangular area, which is then returned as a PIL Image.
"""

import tkinter as tk
import logging
from PIL import Image, ImageGrab

logger = logging.getLogger(__name__)


class SnippingTool:
    """
    Fullscreen overlay that lets the user select a screen region.

    Usage:
        tool = SnippingTool(parent_window)
        image = tool.start()  # Returns PIL Image or None if cancelled
    """

    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.result: Image.Image | None = None

        # Selection coordinates
        self._start_x = 0
        self._start_y = 0
        self._rect_id = None

    def start(self) -> Image.Image | None:
        """
        Minimize the parent, show the overlay, and wait for the user to
        select a region. Returns the captured PIL Image or None if cancelled.
        """
        # Minimize the main app so it doesn't block the screen
        self.parent.iconify()
        self.parent.update()

        # Small delay to let the window minimize and screen settle
        import time
        time.sleep(0.3)

        # Create a fullscreen top-level window
        self.overlay = tk.Toplevel()
        self.overlay.attributes("-fullscreen", True)
        self.overlay.attributes("-topmost", True)
        self.overlay.lift()
        self.overlay.config(cursor="crosshair")

        # Take a screenshot of the entire screen to use as the overlay background
        self._screenshot = ImageGrab.grab()
        self._dim_screenshot = self._screenshot.copy()
        # Darken the screenshot to indicate snipping mode
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(self._dim_screenshot)
        self._dim_screenshot = enhancer.enhance(0.5)

        # Convert to Tk-compatible image
        from PIL import ImageTk
        self._bg_image = ImageTk.PhotoImage(self._dim_screenshot)

        # Canvas filling the entire overlay
        self.canvas = tk.Canvas(
            self.overlay,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._bg_image)

        # Instruction text
        self.canvas.create_text(
            self._dim_screenshot.width // 2,
            30,
            text="Click and drag to select a region  |  Press Escape to cancel",
            fill="white",
            font=("Segoe UI", 14, "bold"),
        )

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.overlay.bind("<Escape>", self._on_cancel)

        # Block until the overlay is closed
        self.parent.wait_window(self.overlay)

        # Restore the main window
        self.parent.deiconify()
        self.parent.update()

        return self.result

    # ── Event handlers ─────────────────────────────────────────────

    def _on_press(self, event: tk.Event):
        self._start_x = event.x
        self._start_y = event.y
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            self._start_x, self._start_y,
            self._start_x, self._start_y,
            outline="red", width=2,
        )

    def _on_drag(self, event: tk.Event):
        if self._rect_id:
            self.canvas.coords(
                self._rect_id,
                self._start_x, self._start_y,
                event.x, event.y,
            )

    def _on_release(self, event: tk.Event):
        x1 = min(self._start_x, event.x)
        y1 = min(self._start_y, event.y)
        x2 = max(self._start_x, event.x)
        y2 = max(self._start_y, event.y)

        # Ignore tiny accidental clicks
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return

        # Crop the *original* (un-dimmed) screenshot
        self.result = self._screenshot.crop((x1, y1, x2, y2))
        logger.info(f"Snipped region: ({x1}, {y1}) -> ({x2}, {y2}), size={self.result.size}")

        self.overlay.destroy()

    def _on_cancel(self, event: tk.Event = None):
        logger.info("Snipping cancelled by user")
        self.result = None
        self.overlay.destroy()
