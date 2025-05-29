#!/usr/bin/env python3
"""
Reverse Archive Search - Main Application

IMPORTANT: Now uses HTML files instead of JSON for fresh Discord URLs!
This solves the 24-hour URL expiration issue.

To use:
1. Export your Discord channel as HTML (much faster than JSON)
2. Launch this application
3. Use the "Load HTML Archive" button to select your HTML file
4. HTML files contain fresh URLs that won't expire for 24 hours!
"""

import sys
import logging
import tkinter as tk
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.config import Config, setup_logging
from gui.main_window import MainWindow

def main():
    """Main application entry point"""
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Reverse Archive Search with HTML File Selection!")
    
    try:
        # Load configuration
        config = Config.load_default()
        
        # Create main window
        root = tk.Tk()
        app = MainWindow(root, config)
        
        logger.info("Launching search interface...")
        
        # Start the application
        root.mainloop()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 