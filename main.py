#!/usr/bin/env python3
"""
Reverse Archive Search - Main application entry point.
"""

import tkinter as tk
import logging
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import Config, setup_logging
from src.gui.main_window import MainWindow

def main():
    """Main application entry point"""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config.load_default()
        
        # Create main window
        root = tk.Tk()
        app = MainWindow(root, config)
        
        logger.info("Starting Reverse Archive Search application")
        logger.info(f"Cache directory: {config.cache.cache_dir}")
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main() 