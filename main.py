#!/usr/bin/env python3
"""
Reverse Archive Search - Main Application
A tool for finding original Discord messages containing similar images using CLIP semantic similarity.
"""

import sys
import logging
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Local imports
from src.gui.main_window import MainWindow
from src.core.logger import setup_logging
from src.core.config import Config

def main():
    """Main application entry point"""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting Reverse Archive Search application")
        
        # Initialize configuration
        config = Config()
        
        # Create main window
        root = tk.Tk()
        app = MainWindow(root, config)
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        # Handle critical errors
        error_msg = f"Critical error starting application: {str(e)}"
        logging.error(error_msg, exc_info=True)
        
        # Show error dialog if possible
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            messagebox.showerror("Application Error", error_msg)
        except:
            print(f"CRITICAL ERROR: {error_msg}")
        
        sys.exit(1)

if __name__ == "__main__":
    main() 