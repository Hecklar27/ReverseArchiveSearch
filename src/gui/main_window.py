"""
Main application window using Tkinter.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import webbrowser
import logging
from pathlib import Path
from typing import Optional, List
import threading
import pickle
from PIL import Image, ImageTk
import requests
from io import BytesIO

from core.config import Config
from data.discord_parser import DiscordParser
from data.models import DiscordMessage, SearchResult, ProcessingStats
from search.strategies import SearchEngine
from gui.snipping_tool import SnippingTool

logger = logging.getLogger(__name__)

class MainWindow:
    """Main application window"""
    
    def __init__(self, root: tk.Tk, config: Config):
        self.root = root
        self.config = config
        self.discord_parser = None  # Will be initialized when needed
        self.search_engine = SearchEngine(config)
        
        # State variables
        self.discord_messages: List[DiscordMessage] = []
        self.current_results: List[SearchResult] = []
        self.user_image_path: Optional[Path] = None
        self.html_archive_path: Optional[Path] = None
        self.selected_result_image: Optional[Image.Image] = None  # Store result image for display
        
        self._setup_window()
        self._create_widgets()
        self._setup_layout()
        
        # Show device info on startup
        self._show_device_info()
        
        # Check cache status on startup
        self._update_cache_status()
        
        # Try to load previously parsed messages
        self._load_previous_parsed_messages()
    
    def _setup_window(self):
        """Configure the main window"""
        self.root.title(self.config.ui.window_title)
        self.root.geometry(self.config.ui.window_size)
        self.root.minsize(800, 600)
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)  # Main content area
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        
        # Control Panel (Top)
        self.control_frame = ttk.Frame(self.root, padding="10")
        
        # File Selection
        ttk.Label(self.control_frame, text="User Image:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.user_image_var = tk.StringVar()
        self.user_image_entry = ttk.Entry(self.control_frame, textvariable=self.user_image_var, width=50)
        self.user_image_entry.grid(row=0, column=1, padx=(5, 0), pady=2, sticky=tk.EW)
        ttk.Button(self.control_frame, text="Browse",
                  command=self._browse_user_image).grid(row=0, column=2, padx=(5, 0), pady=2)
        ttk.Button(self.control_frame, text="Snip Screen",
                  command=self._snip_screen).grid(row=0, column=3, padx=(5, 0), pady=2)
        
        ttk.Label(self.control_frame, text="HTML Archive:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.html_file_var = tk.StringVar()
        self.html_file_entry = ttk.Entry(self.control_frame, textvariable=self.html_file_var, width=50)
        self.html_file_entry.grid(row=1, column=1, padx=(5, 0), pady=2, sticky=tk.EW)
        ttk.Button(self.control_frame, text="Browse", 
                  command=self._browse_html_archive).grid(row=1, column=2, padx=(5, 0), pady=2)
        
        # Search Controls
        search_frame = ttk.Frame(self.control_frame)
        search_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=tk.EW)
        
        # Model Selection
        model_frame = ttk.LabelFrame(search_frame, text="CLIP Model", padding="5")
        model_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # Model dropdown
        self.model_var = tk.StringVar()
        model_options = self.config.clip.get_display_options()
        self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                          values=model_options, state="readonly", width=25)
        self.model_dropdown.pack(anchor=tk.W)
        
        # Set current model
        current_model_display = self.config.clip.get_model_info(self.config.clip.model_name)["display_name"]
        self.model_var.set(current_model_display)
        
        # Model info label
        self.model_info_var = tk.StringVar()
        self.model_info_label = ttk.Label(model_frame, textvariable=self.model_info_var, 
                                         font=('TkDefaultFont', 8), wraplength=200)
        self.model_info_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Update model info
        self._update_model_info()
        
        # Bind model selection change
        self.model_dropdown.bind('<<ComboboxSelected>>', self._on_model_change)
        
        # Real-time search button
        self.search_button = ttk.Button(search_frame, text="Search (Real-time)", 
                                       command=self._start_search)
        self.search_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cache controls (Phase 2)
        cache_frame = ttk.LabelFrame(search_frame, text="Cache Mode", padding="5")
        cache_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cache status
        self.cache_status_var = tk.StringVar(value="No cache")
        self.cache_status_label = ttk.Label(cache_frame, textvariable=self.cache_status_var, 
                                           font=('TkDefaultFont', 8))
        self.cache_status_label.pack(anchor=tk.W)
        
        # Cache controls sub-frame
        cache_controls = ttk.Frame(cache_frame)
        cache_controls.pack(fill=tk.X, pady=(2, 0))
        
        # Pre-process archive button
        self.preprocess_button = ttk.Button(cache_controls, text="Pre-process Archive", 
                                          command=self._start_cache_build, state='disabled')
        self.preprocess_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Cached search button
        self.cached_search_button = ttk.Button(cache_controls, text="Search (Cached)", 
                                             command=self._start_cached_search, state='disabled')
        self.cached_search_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear cache button
        self.clear_cache_button = ttk.Button(cache_controls, text="Clear Cache", 
                                           command=self._clear_cache, state='disabled')
        self.clear_cache_button.pack(side=tk.LEFT)
        
        # Map Art Detection controls (moved to cache section)
        map_art_controls = ttk.Frame(cache_frame)
        map_art_controls.pack(fill=tk.X, pady=(5, 0))
        
        # Main map art detection toggle with info text next to it
        main_detection_frame = ttk.Frame(map_art_controls)
        main_detection_frame.pack(anchor=tk.W)
        
        self.map_art_var = tk.BooleanVar(value=self.config.vision.enable_map_art_detection)
        self.map_art_checkbox = ttk.Checkbutton(main_detection_frame, text="Enable Map Art Detection",
                                               variable=self.map_art_var,
                                               command=self._on_map_art_toggle)
        self.map_art_checkbox.pack(side=tk.LEFT)
        
        # Info text next to main checkbox
        ttk.Label(main_detection_frame, text="💡", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(5, 2))
        self.map_art_info_label = ttk.Label(main_detection_frame, text="Auto-disabled during cache building, usually more accurate", 
                 font=('TkDefaultFont', 8), foreground='gray')
        self.map_art_info_label.pack(side=tk.LEFT)
        
        # Map art status label (below controls)
        self.map_art_status_var = tk.StringVar()
        self.map_art_status_label = ttk.Label(map_art_controls, textvariable=self.map_art_status_var,
                                             font=('TkDefaultFont', 8), wraplength=200)
        self.map_art_status_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Update map art status
        self._update_map_art_status()
        
        # Device info
        self.device_label = ttk.Label(search_frame, text="Device: Loading...")
        self.device_label.pack(side=tk.RIGHT)
        
        # Configure column weight
        self.control_frame.columnconfigure(1, weight=1)
        
        # Progress Bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(self.root, textvariable=self.progress_var)
        
        self.progress_bar = ttk.Progressbar(self.root, mode='determinate')
        
        # Results Area
        self.results_frame = ttk.Frame(self.root)
        
        # Results header
        results_header = ttk.Frame(self.results_frame)
        results_header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_header, text="Search Results:", font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)
        self.results_count_label = ttk.Label(results_header, text="")
        self.results_count_label.pack(side=tk.RIGHT)
        
        # Results listbox with scrollbar
        results_list_frame = ttk.Frame(self.results_frame)
        results_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview for results
        columns = ('similarity', 'author', 'filename', 'timestamp')
        self.results_tree = ttk.Treeview(results_list_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        self.results_tree.heading('similarity', text='Similarity')
        self.results_tree.heading('author', text='Author')
        self.results_tree.heading('filename', text='Filename')
        self.results_tree.heading('timestamp', text='Timestamp')
        
        self.results_tree.column('similarity', width=80, anchor='center')
        self.results_tree.column('author', width=150)
        self.results_tree.column('filename', width=200)
        self.results_tree.column('timestamp', width=150)
        
        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(results_list_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to open Discord link
        self.results_tree.bind('<Double-1>', self._on_result_double_click)
        
        # Results details frame
        details_frame = ttk.LabelFrame(self.results_frame, text="Selected Result Details", padding="10")
        details_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create main details container with two sections
        details_container = ttk.Frame(details_frame)
        details_container.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Text details
        text_details_frame = ttk.Frame(details_container)
        text_details_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.details_text = tk.Text(text_details_frame, height=4, wrap=tk.WORD, state='disabled')
        details_scrollbar = ttk.Scrollbar(text_details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right side: Image preview (always show, adjust behavior based on map detection setting)
        image_preview_frame = ttk.LabelFrame(details_container, text="Image Preview", padding="5")
        image_preview_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Image preview canvas
        self.result_preview_canvas = tk.Canvas(image_preview_frame, width=200, height=150, bg='lightgray')
        self.result_preview_canvas.pack()
        
        # Preview status
        self.preview_status_var = tk.StringVar(value="Select a result to see image preview")
        self.preview_status_label = ttk.Label(image_preview_frame, textvariable=self.preview_status_var, 
                                             font=('TkDefaultFont', 8), wraplength=190)
        self.preview_status_label.pack(pady=(5, 0))
        
        # Initialize with empty preview
        self._clear_result_preview()
        
        # Discord link button (below both sections)
        button_frame = ttk.Frame(details_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.discord_link_button = ttk.Button(button_frame, text="Open in Discord", 
                                            command=self._open_discord_link, state='disabled')
        self.discord_link_button.pack(side=tk.LEFT)
        
        # Show original button (always show if we have preview canvas)
        self.show_original_button = ttk.Button(button_frame, text="Show Original Image", 
                                             command=self._show_original_result_image, state='disabled')
        self.show_original_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Bind result selection
        self.results_tree.bind('<<TreeviewSelect>>', self._on_result_select)
    
    def _setup_layout(self):
        """Setup the main layout"""
        self.control_frame.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=(10, 0))
        self.progress_label.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=(5, 0))
        self.progress_bar.grid(row=2, column=0, sticky=tk.EW, padx=10, pady=(0, 5))
        self.results_frame.grid(row=3, column=0, sticky=tk.NSEW, padx=10, pady=10)
        
        # Hide progress initially
        self.progress_bar.grid_remove()
    
    def _show_device_info(self):
        """Update device information display"""
        try:
            # Get model information
            model_info = self.config.clip.get_model_info(self.config.clip.model_name)
            model_type = self.config.clip.get_model_type(self.config.clip.model_name)
            
            # Get device info from the appropriate engine
            if hasattr(self.search_engine, 'optimized_strategy') and hasattr(self.search_engine.optimized_strategy, 'clip_engine'):
                device_info = self.search_engine.optimized_strategy.clip_engine.get_device_info()
                device_text = device_info.get('device', f'{model_type.upper()} {self.config.clip.model_name}')
                
                # Add performance ratings
                if model_info:
                    speed = model_info.get('speed_rating', 'Unknown')
                    accuracy = model_info.get('accuracy_rating', 'Unknown')
                    device_text += f" | Speed: {speed}, Accuracy: {accuracy}"
            else:
                # Fallback display
                device_text = f"Engine: {model_type.upper()} {self.config.clip.model_name}"
                if model_info:
                    speed = model_info.get('speed_rating', 'Unknown')
                    accuracy = model_info.get('accuracy_rating', 'Unknown')
                    device_text += f" | Speed: {speed}, Accuracy: {accuracy}"
            
            # Show if map art detection is enabled
            if self.config.vision.enable_map_art_detection:
                device_text += " + Map Detection"
            
            self.device_label.config(text=device_text)
            
            # Log detailed info for debugging
            logger.info(f"Engine: {model_type.upper()}")
            logger.info(f"Model: {self.config.clip.model_name}")
            logger.info(f"Performance: {model_info.get('speed_rating', 'Unknown')} speed, {model_info.get('accuracy_rating', 'Unknown')} accuracy")
            
            if hasattr(self.search_engine, 'optimized_strategy'):
                device_info = self.search_engine.optimized_strategy.clip_engine.get_device_info()
                logger.info(f"Device Status: {device_info.get('cuda_status', 'Unknown')}")
                if device_info.get('cuda_error'):
                    logger.warning(f"CUDA Error: {device_info['cuda_error']}")
                
        except Exception as e:
            # Fallback display
            model_name = getattr(self.config.clip, 'model_name', 'Unknown')
            model_type = self.config.clip.get_model_type(model_name)
            self.device_label.config(text=f"Engine: {model_type.upper()} {model_name} (Error)")
            logger.warning(f"Failed to get device info for {model_type}: {e}")
    
    def _browse_user_image(self):
        """Open file dialog to select user image"""
        filetypes = [
            ('Image files', '*.png *.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select User Image",
            filetypes=filetypes
        )
        
        if filename:
            self.user_image_path = Path(filename)
            self.user_image_var.set(str(self.user_image_path))
            logger.info(f"Selected user image: {self.user_image_path}")

    def _snip_screen(self):
        """Launch the snipping tool and use the captured image"""
        import tempfile
        import os

        tool = SnippingTool(self.root)
        image = tool.start()

        if image is None:
            return

        # Save the snipped image to a temp file so the rest of the pipeline
        # (which expects a file path) works without modification.
        tmp_dir = os.path.join(tempfile.gettempdir(), "ReverseArchiveSearch")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, "snip.png")
        image.save(tmp_path, "PNG")

        self.user_image_path = Path(tmp_path)
        self.user_image_var.set(f"[Screen Snip] {self.user_image_path}")
        logger.info(f"Screen snip saved to: {self.user_image_path}")

    def _browse_html_archive(self):
        """Open file dialog to select HTML archive"""
        filetypes = [
            ('HTML files', '*.html'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select HTML Archive",
            filetypes=filetypes
        )
        
        if filename:
            self.html_archive_path = Path(filename)
            self.html_file_var.set(str(self.html_archive_path))
            logger.info(f"Selected HTML archive: {self.html_archive_path}")
            
            # Parse HTML archive in background
            self._parse_html_archive()
    
    def _parse_html_archive(self):
        """Parse HTML archive file in background"""
        if not self.html_archive_path:
            return
            
        # Show progress bar during parsing
        self.progress_bar.grid()
        self.progress_bar.config(value=0, mode='determinate')
        self._update_progress("Starting HTML parsing...")
        
        def progress_callback(current: int, total: int, status: str):
            """Update progress during HTML parsing"""
            progress_percent = (current / total * 100) if total > 0 else 0
            self.root.after(0, lambda: self._update_parsing_progress(progress_percent, status))
        
        def parse_worker():
            try:
                # Create Discord parser for HTML file
                self.discord_parser = DiscordParser(str(self.html_archive_path))
                self.discord_messages = self.discord_parser.parse_messages(progress_callback)
                
                # Update UI in main thread
                self.root.after(0, self._on_discord_parsed)
            except Exception as e:
                error_msg = f"Failed to parse HTML archive: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.root.after(0, lambda: self._on_parsing_error(error_msg))
        
        threading.Thread(target=parse_worker, daemon=True).start()
    
    def _on_discord_parsed(self):
        """Called when HTML archive parsing is complete"""
        # Hide progress bar
        self.progress_bar.grid_remove()
        
        self._update_progress(f"Loaded {len(self.discord_messages)} Discord messages from HTML")
        logger.info(f"Successfully parsed {len(self.discord_messages)} Discord messages from HTML archive")
        
        # Save parsed messages to cache for future use
        if self.html_archive_path and self.discord_messages:
            try:
                cache_manager = self.search_engine.cache_manager
                cache_manager.save_parsed_messages(self.discord_messages, str(self.html_archive_path))
                logger.info("Saved parsed messages to cache for future use")
            except Exception as e:
                logger.warning(f"Failed to save parsed messages to cache: {e}")
        
        # Update cache status when new data is loaded
        self._update_cache_status()
    
    def _start_search(self):
        """Start the image search process"""
        # Validate inputs
        if not self.user_image_path or not self.user_image_path.exists():
            self._show_error("Please select a valid user image")
            return
            
        if not self.discord_messages:
            self._show_error("Please select and load an HTML archive file")
            return
        
        # Check image format
        if not self.config.is_supported_image(self.user_image_path):
            self._show_error(f"Unsupported image format. Supported: .png, .jpg, .jpeg")
            return
        
        # Disable search button and show progress
        self.search_button.config(state='disabled')
        self.progress_bar.grid()
        self.progress_bar['value'] = 0
        
        # Clear previous results
        self._clear_results()
        
        # Start search in background thread - real-time search
        
        def progress_callback(current: int, total: int, status: str):
            """Progress callback for search updates"""
            progress_percent = (current / total * 100) if total > 0 else 0
            progress_text = f"{status} ({current}/{total} images)"
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_search_progress(progress_percent, progress_text))
        
        def search_worker():
            try:
                self._update_progress("Starting search...")
                
                # Update map preview to show actual image that will be used
                # self.root.after(0, self._update_map_preview_during_search)
                
                results, stats = self.search_engine.search(
                    self.user_image_path, 
                    self.discord_messages, 
                    use_cache=False,  # Real-time search
                    progress_callback=progress_callback
                )
                
                # Update UI in main thread
                self.root.after(0, lambda: self._on_search_complete(results, stats))
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Search failed: {e}")
                self.root.after(0, lambda: self._on_search_error(error_message))
        
        threading.Thread(target=search_worker, daemon=True).start()
    
    def _on_search_complete(self, results: List[SearchResult], stats: ProcessingStats):
        """Called when search is complete"""
        self.current_results = results
        
        # Stop progress
        self.progress_bar['value'] = 100
        self.progress_bar.grid_remove()
        
        # Re-enable all buttons
        self.search_button.config(state='normal')
        self.cached_search_button.config(state='normal')
        self.preprocess_button.config(state='normal')
        
        # Update results display
        self._display_results(results, stats)
        
        # Check for expired links and show warning
        self._check_expired_links_warning(stats)
        
        # Update progress text
        self._update_progress(f"Search complete - found {len(results)} results in {stats.processing_time_seconds:.1f}s")
    
    def _check_expired_links_warning(self, stats: ProcessingStats):
        """Check if there are significant expired links and show warning to user"""
        if stats.expired_links > 0:
            expired_percentage = (stats.expired_links / stats.total_images) * 100 if stats.total_images > 0 else 100
            
            # Log the statistics for debugging
            logger.info(f"Expired links check: {stats.expired_links} expired out of {stats.total_images} total ({expired_percentage:.1f}%)")
            
            # Show warning if more than 10% of links are expired, or if more than 50 links are expired
            # Special case: if total_images is 0 but we have expired_links, show warning anyway
            if expired_percentage > 10 or stats.expired_links > 50 or (stats.total_images == 0 and stats.expired_links > 0):
                warning_message = (
                    f"⚠️ HTML Archive Contains Expired Links!\n\n"
                    f"Found {stats.expired_links} expired links out of {stats.total_images} total images "
                    f"({expired_percentage:.1f}%).\n\n"
                    f"This means your HTML export file is old and Discord links have expired.\n\n"
                    f"To get the latest results:\n"
                    f"1. Re-export the mapart-archive channel using DiscordChatExporter\n"
                    f"2. Load the new HTML file\n"
                    f"3. Clear and rebuild the cache if using cached search\n\n"
                    f"Note: Discord attachment links expire after 24 hours."
                )
                
                messagebox.showwarning("Expired Links Detected", warning_message)
                logger.warning(f"Expired links warning shown to user: {stats.expired_links}/{stats.total_images} links expired")
            elif stats.expired_links > 0:
                # Minor warning in progress text only
                logger.info(f"Minor expired links detected: {stats.expired_links} expired links")
    
    def _on_search_error(self, error_message: str):
        """Handle search error"""
        self.progress_bar.grid_remove()
        self.progress_var.set("Search failed")
        
        # Re-enable all buttons
        self.search_button.config(state='normal')
        self.cached_search_button.config(state='normal')
        self.preprocess_button.config(state='normal')
        
        self._show_error(f"Search failed: {error_message}")
        logger.error(f"Search error: {error_message}")
    
    def _update_search_progress(self, progress_percent: float, progress_text: str):
        """Update search progress"""
        self.progress_bar.stop()  # Stop indeterminate mode
        self.progress_bar.config(mode='determinate', value=progress_percent)
        self.progress_var.set(progress_text)
        self.root.update_idletasks()
    
    def _display_results(self, results: List[SearchResult], stats: ProcessingStats):
        """Display search results in the tree view"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add results
        for i, result in enumerate(results):
            author_name = result.message.author.nickname or result.message.author.name
            # Convert cosine similarity (-1 to 1) to percentage (0% to 100%)
            # Map: -1 -> 0%, 0 -> 50%, 1 -> 100%
            similarity_pct = f"{(result.similarity_score + 1) * 50:.1f}%"
            
            # Format timestamp
            timestamp = result.message.timestamp[:10] if result.message.timestamp else "Unknown"
            
            self.results_tree.insert('', 'end', values=(
                similarity_pct,
                author_name,
                result.attachment.filename,
                timestamp
            ))
        
        # Update results count with expired links info if any
        results_text = f"{len(results)} results"
        if stats.expired_links > 0:
            results_text += f" ({stats.expired_links} expired links)"
        self.results_count_label.config(text=results_text)
        
        # Update statistics in progress with expired links info
        success_rate = stats.get_success_rate()
        progress_text = (
            f"Processed {stats.processed_images}/{stats.total_images} images "
            f"({success_rate:.1f}% success) in {stats.processing_time_seconds:.1f}s"
        )
        
        if stats.expired_links > 0:
            progress_text += f" - {stats.expired_links} expired links detected"
        
        self._update_progress(progress_text)
    
    def _on_result_select(self, event):
        """Handle result selection"""
        selection = self.results_tree.selection()
        if not selection:
            return
            
        # Get selected item index
        item = selection[0]
        index = self.results_tree.index(item)
        
        if index < len(self.current_results):
            result = self.current_results[index]
            self._show_result_details(result)
    
    def _show_result_details(self, result: SearchResult):
        """Show details for selected result"""
        # Enable text widget
        self.details_text.config(state='normal')
        
        # Clear and update content
        self.details_text.delete(1.0, tk.END)
        
        details = f"Author: {result.message.author.nickname or result.message.author.name}\n"
        # Convert cosine similarity (-1 to 1) to percentage (0% to 100%)
        details += f"Similarity: {(result.similarity_score + 1) * 50:.1f}%\n"
        details += f"Timestamp: {result.message.timestamp}\n"
        details += f"Content: {result.message.content[:200]}{'...' if len(result.message.content) > 200 else ''}\n"
        details += f"Discord URL: {result.discord_url}"
        
        self.details_text.insert(1.0, details)
        
        # Disable text widget
        self.details_text.config(state='disabled')
        
        # Enable Discord link button
        self.discord_link_button.config(state='normal')
        
        # Enable show original button (always enable since we always have preview now)
        self.show_original_button.config(state='normal')
        
        # Load and preview the result image
        self._load_result_image_preview(result)
    
    def _load_result_image_preview(self, result: SearchResult):
        """Load and show the image preview - cropped if map art detection enabled, original if disabled"""
        if not hasattr(self, 'result_preview_canvas'):
            return
        
        def load_worker():
            """Background worker to download and process the image"""
            try:
                # Update status
                self.root.after(0, lambda: self.preview_status_var.set("Downloading image..."))
                
                # Download the image
                response = requests.get(result.attachment.url, timeout=10)
                response.raise_for_status()
                
                # Load image
                original_image = Image.open(BytesIO(response.content)).convert('RGB')
                
                if self.config.vision.enable_map_art_detection:
                    # Map art detection enabled - show cropped version
                    self.root.after(0, lambda: self.preview_status_var.set("Processing map detection..."))
                    
                    # Apply map art detection using the current fast detection setting
                    cropped_images = self._apply_map_art_detection(original_image)
                    
                    if cropped_images:
                        # Use the first (largest) cropped image - same as search engines
                        cropped_image = cropped_images[0]
                        
                        # Store both images for toggling
                        self.selected_result_image = cropped_image
                        self.original_result_image = original_image
                        
                        # Display cropped version
                        self.root.after(0, lambda: self._display_result_preview(cropped_image, True, len(cropped_images)))
                        
                    else:
                        # No map art detected
                        if self.config.vision.fallback_to_full_image:
                            # Store original image
                            self.selected_result_image = original_image
                            self.original_result_image = original_image
                            
                            # Display original
                            self.root.after(0, lambda: self._display_result_preview(original_image, False, 0))
                        else:
                            # Clear preview
                            self.root.after(0, lambda: self._clear_result_preview())
                            self.root.after(0, lambda: self.preview_status_var.set("No map art detected, image skipped"))
                else:
                    # Map art detection disabled - show original image
                    self.root.after(0, lambda: self.preview_status_var.set("Showing original image..."))
                    
                    # Store original image
                    self.selected_result_image = original_image
                    self.original_result_image = original_image
                    
                    # Display original image with different status
                    self.root.after(0, lambda: self._display_result_preview(original_image, False, 0, is_detection_disabled=True))
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download result image: {e}")
                self.root.after(0, lambda: self._clear_result_preview())
                self.root.after(0, lambda: self.preview_status_var.set("Failed to download image"))
                
            except Exception as e:
                logger.error(f"Failed to process result image: {e}")
                self.root.after(0, lambda: self._clear_result_preview())
                self.root.after(0, lambda: self.preview_status_var.set(f"Processing failed: {str(e)[:50]}..."))
        
        # Start download in background
        threading.Thread(target=load_worker, daemon=True).start()
    
    def _apply_map_art_detection(self, image: Image.Image):
        """Apply map art detection to an image using current engine settings"""
        try:
            # Import map art detector
            from vision.map_art_detector import create_map_art_detector
            
            # Create detector with current settings
            detector = create_map_art_detector(
                method=self.config.vision.detection_method,
                use_fast_detection=self.config.vision.use_fast_detection
            )
            
            # Process image
            cropped_images = detector.process_image(image)
            return cropped_images
            
        except ImportError:
            logger.warning("Map art detection module not available")
            return []
        except Exception as e:
            logger.error(f"Map art detection failed: {e}")
            return []
    
    def _display_result_preview(self, image: Image.Image, is_cropped: bool, num_regions: int, is_detection_disabled: bool = False):
        """Display a result image in the preview canvas"""
        try:
            # Calculate display size while maintaining aspect ratio
            canvas_width = 200
            canvas_height = 150
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Calculate scaling factor
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y)
            
            # Calculate display dimensions
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)
            
            # Resize image for display
            display_image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Clear canvas and display image
            self.result_preview_canvas.delete("all")
            
            # Center the image on canvas
            x_offset = (canvas_width - display_width) // 2
            y_offset = (canvas_height - display_height) // 2
            
            self.result_preview_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
            
            # Keep a reference to prevent garbage collection
            self.result_preview_canvas.image = photo
            
            # Add border indicator and status based on mode
            if is_detection_disabled:
                # Blue border for when detection is disabled - showing original
                self.result_preview_canvas.create_rectangle(x_offset-2, y_offset-2, 
                                                          x_offset+display_width+2, y_offset+display_height+2,
                                                          outline='blue', width=2)
                status_text = "🖼️ Original image (detection disabled)"
            elif is_cropped and num_regions > 0:
                # Green border for cropped map art
                self.result_preview_canvas.create_rectangle(x_offset-2, y_offset-2, 
                                                          x_offset+display_width+2, y_offset+display_height+2,
                                                          outline='green', width=2)
                status_text = f"✅ Cropped map art ({num_regions} region{'s' if num_regions > 1 else ''} detected)"
            else:
                # Orange border for full image (detection enabled but no map found)
                self.result_preview_canvas.create_rectangle(x_offset-2, y_offset-2, 
                                                          x_offset+display_width+2, y_offset+display_height+2,
                                                          outline='orange', width=2)
                if is_cropped:
                    status_text = "⚠️ Using full image (no map art detected)"
                else:
                    status_text = "📋 Showing original image"
            
            # Update status
            self.preview_status_var.set(status_text)
            
        except Exception as e:
            logger.error(f"Failed to display result preview: {e}")
            self._clear_result_preview()
    
    def _clear_result_preview(self):
        """Clear the result preview canvas"""
        if hasattr(self, 'result_preview_canvas') and self.result_preview_canvas:
            self.result_preview_canvas.delete("all")
            self.result_preview_canvas.create_text(100, 75, text="No preview", 
                                                  fill='gray', font=('TkDefaultFont', 10))
    
    def _show_original_result_image(self):
        """Toggle between showing cropped and original version of result image"""
        if not hasattr(self, 'original_result_image') or not self.original_result_image:
            return
        
        try:
            # Check current button text to determine what to show
            current_text = self.show_original_button.cget('text')
            
            if not self.config.vision.enable_map_art_detection:
                # When map art detection is disabled, we're always showing the original
                # The button doesn't need to toggle anything, just inform the user
                messagebox.showinfo("Image Preview", 
                                  "Map art detection is disabled. The preview always shows the original image.\n\n"
                                  "To see cropped map art, enable map art detection in the settings.")
                return
            
            if current_text == "Show Original Image":
                # Currently showing cropped, switch to original
                self._display_result_preview(self.original_result_image, False, 0)
                self.show_original_button.config(text="Show Cropped Map")
                
            else:
                # Currently showing original, switch to cropped
                if hasattr(self, 'selected_result_image') and self.selected_result_image:
                    # Determine if this is actually cropped or just the same as original
                    if self.selected_result_image.size != self.original_result_image.size:
                        self._display_result_preview(self.selected_result_image, True, 1)
                    else:
                        self._display_result_preview(self.selected_result_image, False, 0)
                self.show_original_button.config(text="Show Original Image")
                
        except Exception as e:
            logger.error(f"Failed to toggle result image: {e}")
    
    def _on_result_double_click(self, event):
        """Handle double-click on result"""
        self._open_discord_link()
    
    def _open_discord_link(self):
        """Open Discord link in browser"""
        selection = self.results_tree.selection()
        if not selection:
            return
            
        item = selection[0]
        index = self.results_tree.index(item)
        
        if index < len(self.current_results):
            result = self.current_results[index]
            try:
                webbrowser.open(result.discord_url)
                logger.info(f"Opened Discord link: {result.discord_url}")
            except Exception as e:
                logger.error(f"Failed to open Discord link: {e}")
                self._show_error(f"Failed to open Discord link:\n{str(e)}")
    
    def _clear_results(self):
        """Clear all results"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        self.results_count_label.config(text="")
        self.details_text.config(state='normal')
        self.details_text.delete(1.0, tk.END)
        self.details_text.config(state='disabled')
        self.discord_link_button.config(state='disabled')
        
        # Disable show original button (now always present)
        self.show_original_button.config(state='disabled')
        
        # Clear result preview (now always present)
        self._clear_result_preview()
        self.preview_status_var.set("Select a result to see image preview")
        
        self.current_results = []
        
        # Clear stored images
        self.selected_result_image = None
        if hasattr(self, 'original_result_image'):
            self.original_result_image = None
    
    def _update_progress(self, message: str):
        """Update progress message"""
        self.progress_var.set(message)
        logger.info(message)
    
    def _show_error(self, message: str):
        """Show error dialog"""
        messagebox.showerror("Error", message)
        logger.error(message)
    
    def _start_cache_build(self):
        """Start building the embedding cache"""
        if not self.discord_messages:
            self._show_error("Please load HTML archive file first")
            return
        
        # Reset expired link warning flag for this cache build
        if hasattr(self, '_cache_expired_warning_shown'):
            delattr(self, '_cache_expired_warning_shown')
        
        # Store original map art detection setting and force disable for cache building
        self._original_map_art_setting = self.config.vision.enable_map_art_detection
        self.config.vision.enable_map_art_detection = False
        
        # Update UI to show cache building state
        self.map_art_info_label.config(text="Cache building: Map art detection temporarily disabled for speed")
        
        # Disable buttons during cache build
        self.preprocess_button.config(state='disabled')
        self.search_button.config(state='disabled')
        self.cached_search_button.config(state='disabled')
        
        # Show progress
        self.progress_bar.grid()
        self.progress_bar.config(value=0)
        self.progress_var.set("Building cache...")
        
        def progress_callback(current: int, total: int, status: str):
            """Update progress during cache building"""
            if total > 0:
                progress_percent = (current / total) * 100
                self.root.after(0, lambda: self._update_cache_progress(progress_percent, status))
        
        def cache_build_worker():
            """Worker thread for cache building"""
            try:
                success, stats = self.search_engine.build_cache(self.discord_messages, progress_callback)
                self.root.after(0, lambda: self._on_cache_build_complete(success, stats))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._on_cache_build_error(error_msg))
        
        # Start cache building in background thread
        threading.Thread(target=cache_build_worker, daemon=True).start()
    
    def _on_cache_build_complete(self, success: bool, stats: ProcessingStats):
        """Handle cache build completion"""
        self.progress_bar.grid_remove()
        
        # Restore original map art detection setting
        if hasattr(self, '_original_map_art_setting'):
            # Restore the original setting
            self.config.vision.enable_map_art_detection = self._original_map_art_setting
            self.map_art_var.set(self._original_map_art_setting)
            delattr(self, '_original_map_art_setting')
        
        # Restore original info text
        self.map_art_info_label.config(text="Auto-disabled during cache building, usually more accurate")
        
        # Update UI state
        self._update_map_art_status()
        
        if success:
            self.progress_var.set("Cache built successfully")
            self._update_cache_status()
            
            # Check for expired links and show warning
            self._check_expired_links_warning(stats)
            
            messagebox.showinfo("Cache Built", 
                              "Embedding cache built successfully!\nCached searches will now be much faster.")
        else:
            self.progress_var.set("Cache build failed")
            
            # Always check for expired links on failure - they might explain why it failed
            self._check_expired_links_warning(stats)
            
            # Show the error message after the expired links warning (if any)
            self._show_error("Failed to build cache. Check logs for details.")
        
        # Re-enable buttons
        self.preprocess_button.config(state='normal')
        self.search_button.config(state='normal')
        # Note: cached_search_button state is handled by _update_cache_status() above
    
    def _on_cache_build_error(self, error_message: str):
        """Handle cache build error"""
        self.progress_bar.grid_remove()
        self.progress_var.set("Cache build failed")
        
        # Restore original map art detection setting
        if hasattr(self, '_original_map_art_setting'):
            # Restore the original setting
            self.config.vision.enable_map_art_detection = self._original_map_art_setting
            self.map_art_var.set(self._original_map_art_setting)
            delattr(self, '_original_map_art_setting')
        
        # Restore original info text
        self.map_art_info_label.config(text="Auto-disabled during cache building, usually more accurate")
        
        # Update UI state
        self._update_map_art_status()
        
        self.preprocess_button.config(state='normal')
        self.search_button.config(state='normal')
        self._show_error(f"Cache build failed: {error_message}")
        logger.error(f"Cache build error: {error_message}")
    
    def _update_cache_progress(self, progress_percent: float, progress_text: str):
        """Update cache build progress"""
        self.progress_bar.config(value=progress_percent)
        self.progress_var.set(f"Building cache: {progress_text}")
        self.root.update_idletasks()
        
        # Early expired link detection during cache building
        if hasattr(self, '_cache_expired_warning_shown'):
            return  # Already shown warning, don't spam
            
        # Check if progress indicates high expired link rate
        # Look for pattern indicating many failed downloads
        if "batch" in progress_text.lower() and self.search_engine.cache_manager.image_downloader:
            try:
                # Get current download statistics
                download_stats = self.search_engine.cache_manager.image_downloader.get_stats()
                total_attempted = download_stats.get('total_downloads', 0)
                expired_links = download_stats.get('expired_links', 0)
                
                # Only check after we've attempted at least 64 downloads (2 batches) to avoid false positives
                if total_attempted >= 64:
                    expired_percentage = (expired_links / total_attempted) * 100 if total_attempted > 0 else 0
                    
                    # Show early warning if >75% of links are expired
                    if expired_percentage > 75:
                        self._cache_expired_warning_shown = True
                        self._show_early_expired_warning(expired_links, total_attempted, expired_percentage)
                        
            except Exception as e:
                logger.debug(f"Error checking early expired links: {e}")
    
    def _show_early_expired_warning(self, expired_links: int, total_attempted: int, expired_percentage: float):
        """Show early warning when majority of links are expired during cache building"""
        warning_message = (
            f"⚠️ High Rate of Expired Links Detected!\n\n"
            f"During cache building, {expired_links} out of {total_attempted} links have expired "
            f"({expired_percentage:.1f}%).\n\n"
            f"This suggests your HTML export file is old and most Discord links have expired.\n\n"
            f"Consider stopping the cache build and:\n"
            f"1. Re-export the mapart-archive channel using DiscordChatExporter\n"
            f"2. Load the new HTML file\n"
            f"3. Restart the cache building process\n\n"
            f"Note: Discord attachment links expire after 24 hours.\n\n"
            f"Continue building cache with expired links?"
        )
        
        # Use askquestion instead of showwarning so user can choose to continue or stop
        result = messagebox.askyesno("Expired Links Detected", warning_message)
        
        if not result:
            # User chose to stop cache building
            logger.info("User chose to stop cache building due to expired links")
            # Note: We can't easily stop the background thread from here, 
            # but the user can manually stop by closing the app or using clear cache
            messagebox.showinfo("Cache Build Continuing", 
                              "Cache build is continuing in background. "
                              "You can stop it by closing the application or using 'Clear Cache' when complete.")
        else:
            logger.info("User chose to continue cache building despite expired links")
    
    def _start_cached_search(self):
        """Start cached search"""
        if not self.user_image_path:
            self._show_error("Please select a user image first")
            return
        
        if not self.discord_messages:
            self._show_error("Please load HTML archive file first")
            return
        
        # Enhanced cache validation with in-memory fallback
        cache_available = False
        
        # First, try standard cache validation
        if self.search_engine.has_cache():
            cache_available = True
            logger.info("Cache available via standard validation")
        else:
            # Fallback: check if cache is available in memory (newly built)
            cache_manager = self.search_engine.cache_manager
            if (hasattr(cache_manager, '_embeddings') and cache_manager._embeddings is not None and
                hasattr(cache_manager, '_metadata') and cache_manager._metadata is not None):
                cache_available = True
                logger.info("Cache available via in-memory fallback - using newly built cache")
            else:
                logger.info("No cache available via any method")
        
        if not cache_available:
            self._show_error("No valid cache available. Please run 'Pre-process Archive' first.")
            return
        
        # Disable buttons during search
        self.cached_search_button.config(state='disabled')
        self.search_button.config(state='disabled')
        self.preprocess_button.config(state='disabled')
        
        # Clear previous results
        self._clear_results()
        
        # Show progress
        self.progress_bar.grid()
        self.progress_bar.config(value=0, mode='indeterminate')
        self.progress_var.set("Starting cached search...")
        self.progress_bar.start()
        
        def progress_callback(current: int, total: int, status: str):
            """Update progress during cached search"""
            if total > 0:
                progress_percent = (current / total) * 100
                self.root.after(0, lambda: self._update_search_progress(progress_percent, status))
        
        def search_worker():
            """Worker thread for cached search"""
            try:
                # Update map preview to show actual image that will be used
                # self.root.after(0, self._update_map_preview_during_search)
                
                results, stats = self.search_engine.search(
                    self.user_image_path, 
                    self.discord_messages, 
                    use_cache=True,
                    progress_callback=progress_callback
                )
                self.root.after(0, lambda: self._on_search_complete(results, stats))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._on_cached_search_error(error_msg))
        
        # Start search in background thread
        threading.Thread(target=search_worker, daemon=True).start()
    
    def _on_cached_search_error(self, error_message: str):
        """Handle cached search error"""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.progress_var.set("Cached search failed")
        
        # Re-enable buttons
        self.cached_search_button.config(state='normal')
        self.search_button.config(state='normal')
        self.preprocess_button.config(state='normal')
        
        self._show_error(f"Cached search failed: {error_message}")
        logger.error(f"Cached search error: {error_message}")
    
    def _clear_cache(self):
        """Clear the embedding cache with model-specific options"""
        try:
            # Get all cache info to show options
            all_cache_info = self.search_engine.get_all_cache_info()
            all_models = all_cache_info['all_models']
            current_model = all_cache_info['current_model']
            
            # Check which models have caches
            cached_models = [model for model, info in all_models.items() 
                           if info.get('status') == 'valid']
            
            if not cached_models:
                messagebox.showinfo("No Cache", "No caches found to clear.")
                return
            
            # Create options dialog
            if len(cached_models) == 1:
                # Only one cache, simple confirmation
                model_name = cached_models[0]
                if messagebox.askyesno("Clear Cache", 
                                      f"Are you sure you want to clear the {model_name} cache?\n"
                                      f"This will remove all pre-computed embeddings for this model."):
                    self.search_engine.clear_cache_for_model(model_name)
                    self._update_cache_status()
                    messagebox.showinfo("Cache Cleared", f"{model_name} cache cleared successfully.")
                    self.progress_var.set(f"{model_name} cache cleared")
            else:
                # Multiple caches, show options
                result = messagebox.askyesnocancel("Clear Cache", 
                                                 f"Multiple caches found:\n"
                                                 f"• {cached_models[0]} cache\n"
                                                 f"• {cached_models[1]} cache\n\n"
                                                 f"Yes = Clear current model ({current_model}) cache\n"
                                                 f"No = Clear all model caches\n"
                                                 f"Cancel = Don't clear anything")
                
                if result is True:
                    # Clear current model cache
                    self.search_engine.clear_cache_for_model(current_model)
                    self._update_cache_status()
                    messagebox.showinfo("Cache Cleared", f"{current_model} cache cleared successfully.")
                    self.progress_var.set(f"{current_model} cache cleared")
                    
                elif result is False:
                    # Clear all caches
                    cleared_models = []
                    for model_name in cached_models:
                        try:
                            self.search_engine.clear_cache_for_model(model_name)
                            cleared_models.append(model_name)
                        except Exception as e:
                            logger.error(f"Failed to clear cache for {model_name}: {e}")
                    
                    self._update_cache_status()
                    if cleared_models:
                        messagebox.showinfo("Caches Cleared", 
                                          f"Cleared caches for: {', '.join(cleared_models)}")
                        self.progress_var.set(f"All caches cleared")
                    
        except Exception as e:
            self._show_error(f"Failed to clear cache: {e}")
    
    def _update_cache_status(self):
        """Update cache status display"""
        try:
            # Get cache info for all models
            all_cache_info = self.search_engine.get_all_cache_info()
            current_model = all_cache_info['current_model']
            all_models = all_cache_info['all_models']
            
            # Get current model cache status
            current_cache = all_models.get(current_model, {'status': 'no_cache'})
            current_available = current_cache.get('status') == 'valid'
            
            # Build status text
            if current_available:
                # Show current model cache status
                status_text = f"✓ {current_model} cache ready"
                
                # Add size info if available
                if 'embeddings_size_mb' in current_cache:
                    status_text += f" ({current_cache['embeddings_size_mb']:.1f}MB)"
                
                # Check other model cache status
                other_models = [m for m in all_models.keys() if m != current_model]
                if other_models:
                    other_model = other_models[0]
                    other_cache = all_models.get(other_model, {'status': 'no_cache'})
                    if other_cache.get('status') == 'valid':
                        status_text += f" | ✓ {other_model} cached"
                    else:
                        status_text += f" | ○ {other_model} no cache"
                
                self.cache_status_var.set(status_text)
                self.cached_search_button.config(state='normal')
                self.clear_cache_button.config(state='normal')
                
            else:
                # No cache for current model
                status_text = f"○ {current_model} no cache"
                
                # Check if other models have cache
                other_models = [m for m in all_models.keys() if m != current_model]
                if other_models:
                    other_model = other_models[0]
                    other_cache = all_models.get(other_model, {'status': 'no_cache'})
                    if other_cache.get('status') == 'valid':
                        status_text += f" | ✓ {other_model} cached"
                    else:
                        status_text += f" | ○ {other_model} no cache"
                
                self.cache_status_var.set(status_text)
                self.cached_search_button.config(state='disabled')
                self.clear_cache_button.config(state='disabled')
            
            # Enable pre-process button if we have Discord data
            if self.discord_messages:
                self.preprocess_button.config(state='normal')
            else:
                self.preprocess_button.config(state='disabled')
                
        except Exception as e:
            logger.warning(f"Failed to update cache status: {e}")
            self.cache_status_var.set("Cache status error")
            self.cached_search_button.config(state='disabled')
            self.clear_cache_button.config(state='disabled')
    
    def _update_parsing_progress(self, progress_percent: float, progress_text: str):
        """Update HTML parsing progress"""
        self.progress_bar.config(value=progress_percent)
        self.progress_var.set(f"Parsing HTML: {progress_text}")
        self.root.update_idletasks()
    
    def _on_parsing_error(self, error_msg: str):
        """Handle HTML parsing error"""
        self.progress_bar.grid_remove()
        self.progress_var.set("Parsing failed")
        self._show_error(error_msg)
    
    def _load_previous_parsed_messages(self):
        """Try to load previously parsed messages"""
        try:
            cache_manager = self.search_engine.cache_manager
            current_model = self.config.clip.model_name
            
            # Get the correct paths for the current model
            cache_paths = cache_manager._get_model_cache_paths(current_model)
            parsed_messages_file = cache_paths['parsed_messages_file']
            parsed_metadata_file = cache_paths['parsed_metadata_file']
            
            # Check if we have parsed messages (cache manager will check other models too)
            # First, we need to find any HTML file that has parsed messages
            html_file_to_check = None
            
            # Try to find an HTML file from any model's cache
            all_models = ['ViT-L/14', 'RN50x64', 'ViT-B/16', 'DINOv2-Base']
            for model in all_models:
                model_paths = cache_manager._get_model_cache_paths(model)
                if model_paths['parsed_metadata_file'].exists():
                    try:
                        with open(model_paths['parsed_metadata_file'], 'rb') as f:
                            metadata = pickle.load(f)
                        html_file_to_check = metadata.html_file_path
                        break
                    except:
                        continue
            
            if html_file_to_check is None:
                logger.info(f"No previously parsed messages found for {current_model}")
                return
            
            # Check if this HTML file is still valid (cache manager handles cross-model checking)
            if cache_manager.has_parsed_messages(html_file_to_check):
                logger.info(f"Loading previously parsed messages for {html_file_to_check}")
                
                # Load the messages (cache manager will load from any model and copy if needed)
                messages = cache_manager.load_parsed_messages()
                if messages:
                    self.discord_messages = messages
                    self.html_archive_path = Path(html_file_to_check)
                    self.html_file_var.set(str(self.html_archive_path))
                    
                    # Update the UI
                    self._update_progress(f"Loaded {len(messages)} Discord messages from previous session")
                    logger.info(f"Successfully loaded {len(messages)} previously parsed messages")
                    
                    # Update cache status to enable buttons
                    self._update_cache_status()
                else:
                    logger.warning("Failed to load parsed messages despite validation passing")
            else:
                logger.info("Previously parsed messages are no longer valid (HTML file changed or missing)")
                # Clean up invalid cached messages
                cache_manager.clear_parsed_messages()
                
        except Exception as e:
            logger.warning(f"Failed to load previously parsed messages: {e}")
            # Clean up potentially corrupted cache
            try:
                cache_manager.clear_parsed_messages()
            except:
                pass

    def _update_model_info(self):
        """Update model information display"""
        try:
            # Get current model info
            model_info = self.config.clip.get_model_info(self.config.clip.model_name)
            
            # Update model info label
            info_text = f"{model_info.get('description', 'No description')}"
            self.model_info_var.set(info_text)
            
        except Exception as e:
            logger.warning(f"Failed to update model info: {e}")
            self.model_info_var.set("Model: Error")
    
    def _on_model_change(self, event):
        """Handle model selection change"""
        try:
            selected_display = self.model_var.get()
            new_model_name = self.config.clip.model_name_from_display(selected_display)
            
            if new_model_name == self.config.clip.model_name:
                return  # No change
            
            logger.info(f"User selected model: {new_model_name}")
            
            # Show loading message
            self.progress_var.set(f"Switching to {new_model_name}...")
            self.root.update_idletasks()
            
            # Switch model in search engine
            self.search_engine.switch_model(new_model_name)
            
            # Update model info display
            self._update_model_info()
            
            # Update device info to reflect new model
            self._show_device_info()
            
            # Update cache status for new model
            self._update_cache_status()
            
            # Update map art status to show model-specific recommendations
            self._update_map_art_status()
            
            self.progress_var.set(f"Switched to {new_model_name}")
            logger.info(f"Successfully switched to model {new_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            self._show_error(f"Failed to switch model: {str(e)}")
            
            # Revert dropdown selection on error
            current_display = self.config.clip.get_model_info(self.config.clip.model_name)["display_name"]
            self.model_var.set(current_display)
    
    def _on_map_art_toggle(self):
        """Handle map art detection toggle"""
        try:
            # Update the configuration
            old_state = self.config.vision.enable_map_art_detection
            new_state = self.map_art_var.get()
            self.config.vision.enable_map_art_detection = new_state
            
            # Reinitialize the search engine's CLIP engine to apply the new setting
            if old_state != new_state:
                # Show progress and disable the checkbox during reinitialization
                self.map_art_checkbox.config(state='disabled')
                self.progress_var.set("Updating map art detection...")
                self.progress_bar.grid()
                self.progress_bar.config(mode='indeterminate')
                self.progress_bar.start()
                self.root.update_idletasks()
                
                def reinitialize_worker():
                    """Background worker to reinitialize search engine"""
                    try:
                        # Properly dispose of old search engine using its cleanup method
                        if hasattr(self, 'search_engine') and self.search_engine:
                            try:
                                logger.info("Cleaning up old SearchEngine before creating new one")
                                self.search_engine.cleanup()
                                del self.search_engine
                                logger.info("Old SearchEngine disposed of successfully")
                            except Exception as e:
                                logger.warning(f"Error disposing old SearchEngine: {e}")
                        
                        # Additional GPU cache clearing
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                logger.info("Additional GPU cache cleared before creating new SearchEngine")
                        except Exception as e:
                            logger.warning(f"Could not clear GPU cache: {e}")
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Now create the new search engine
                        logger.info(f"Creating new SearchEngine with map art detection {'enabled' if new_state else 'disabled'}")
                        new_search_engine = SearchEngine(self.config)
                        
                        # Prepare status information in background thread
                        try:
                            # Get device info
                            device_info = new_search_engine.optimized_strategy.clip_engine.get_device_info()
                            model_info = self.config.clip.get_model_info(self.config.clip.model_name)
                            
                            device_text = f"Engine: CLIP {self.config.clip.model_name} ({device_info['device'].upper()})"
                            
                            if device_info.get('using_cuda', False):
                                gpu_name = device_info.get('gpu_name', 'Unknown GPU')
                                gpu_memory = device_info.get('gpu_memory_gb', 'Unknown')
                                device_text += f" - {gpu_name}, {gpu_memory}GB"
                            elif device_info.get('gpu_available', False):
                                gpu_name = device_info.get('gpu_name', 'Unknown GPU')
                                device_text += f" - GPU available: {gpu_name}"
                            else:
                                device_text += " - No CUDA"
                            
                            if model_info:
                                speed_rating = model_info.get('speed_rating', 'Unknown')
                                accuracy_rating = model_info.get('accuracy_rating', 'Unknown')
                                device_text += f" | Speed: {speed_rating}, Accuracy: {accuracy_rating}"
                            
                            if self.config.vision.enable_map_art_detection:
                                device_text += " + Map Detection"
                                
                        except Exception as e:
                            logger.warning(f"Failed to get device info: {e}")
                            device_text = f"Engine: CLIP {self.config.clip.model_name} (Error)"
                        
                        # Prepare map art status
                        try:
                            if self.config.vision.enable_map_art_detection:
                                current_model = self.config.clip.model_name
                                if current_model == "ViT-L/14":
                                    map_art_status = "Enabled - Recommended with ViT-L/14. Auto-disabled during cache building."
                                else:
                                    map_art_status = "Enabled - Usually more accurate."
                            else:
                                map_art_status = "Disabled - Uses full images for similarity search"
                        except Exception as e:
                            logger.error(f"Failed to prepare map art status: {e}")
                            map_art_status = "Status: Error"
                        
                        # Update in main thread with prepared data
                        def update_ui():
                            try:
                                # Replace the search engine
                                self.search_engine = new_search_engine
                                
                                # Update status displays with pre-computed values
                                self.device_label.config(text=device_text)
                                self.map_art_status_var.set(map_art_status)
                                
                                # Update cache status (this should be lightweight)
                                self._update_cache_status_lightweight()
                                
                                # Hide progress and re-enable checkbox
                                self.progress_bar.stop()
                                self.progress_bar.grid_remove()
                                self.map_art_checkbox.config(state='normal')
                                self.progress_var.set("Ready")
                                
                                logger.info(f"Map art detection {'enabled' if new_state else 'disabled'}")
                                
                                # Show info message about the change (defer to avoid blocking)
                                self.root.after(100, lambda: self._show_map_art_toggle_message(new_state))
                                
                            except Exception as e:
                                logger.error(f"Failed to update UI after map art toggle: {e}")
                                self._handle_toggle_error(old_state, str(e))
                        
                        self.root.after(0, update_ui)
                        
                    except Exception as e:
                        logger.error(f"Failed to reinitialize search engine: {e}")
                        error_msg = str(e)
                        self.root.after(0, lambda: self._handle_toggle_error(old_state, error_msg))
                
                # Start reinitialization in background thread
                threading.Thread(target=reinitialize_worker, daemon=True).start()
                
        except Exception as e:
            logger.error(f"Failed to toggle map art detection: {e}")
            self._handle_toggle_error(old_state, str(e))

    def _handle_toggle_error(self, old_state: bool, error_message: str):
        """Handle errors during map art detection toggle"""
        try:
            # Hide progress and re-enable checkbox
            self.progress_bar.stop()
            self.progress_bar.grid_remove()
            self.map_art_checkbox.config(state='normal')
            
            # Revert the checkbox state on error
            self.map_art_var.set(old_state)
            self.config.vision.enable_map_art_detection = old_state
            
            # Update status
            self._update_map_art_status()
            self.progress_var.set("Ready")
            
            # Show error message
            self._show_error(f"Failed to toggle map art detection: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to handle toggle error: {e}")
            self.progress_var.set("Error")

    def _update_map_art_status(self):
        """Update map art detection status display"""
        try:
            if self.config.vision.enable_map_art_detection:
                # Get current model for recommendation
                current_model = self.config.clip.model_name
                
                if current_model == "ViT-L/14":
                    status_text = "Enabled - Recommended with ViT-L/14. Auto-disabled during cache building."
                else:
                    status_text = "Enabled - Usually more accurate."
            else:
                status_text = "Disabled - Uses full images for similarity search"
            self.map_art_status_var.set(status_text)
        except Exception as e:
            logger.error(f"Failed to update map art detection status: {e}")
            self.map_art_status_var.set("Status: Error")
            
    def _update_preview_visibility(self):
        """Update visibility of preview elements based on map art detection setting"""
        try:
            # Note: For now, we keep the preview elements as they are created at startup
            # In a future enhancement, we could dynamically show/hide preview elements
            # This method is prepared for that enhancement
            pass
        except Exception as e:
            logger.error(f"Failed to update preview visibility: {e}")

    def _update_cache_status_lightweight(self):
        """Lightweight cache status update that doesn't do heavy operations"""
        try:
            # Quick check if we have cache in memory
            cache_manager = self.search_engine.cache_manager
            current_model = self.config.clip.model_name
            
            if (hasattr(cache_manager, '_embeddings') and cache_manager._embeddings is not None and
                hasattr(cache_manager, '_metadata') and cache_manager._metadata is not None):
                # Cache available in memory
                status_text = f"✓ {current_model} cache ready"
                self.cache_status_var.set(status_text)
                self.cached_search_button.config(state='normal')
                self.clear_cache_button.config(state='normal')
            else:
                # No cache in memory, would need file check (defer this)
                status_text = f"○ {current_model} checking cache..."
                self.cache_status_var.set(status_text)
                self.cached_search_button.config(state='disabled')
                self.clear_cache_button.config(state='disabled')
                
                # Schedule full cache check for later
                self.root.after(500, self._update_cache_status)
            
            # Enable pre-process button if we have Discord data
            if self.discord_messages:
                self.preprocess_button.config(state='normal')
            else:
                self.preprocess_button.config(state='disabled')
                
        except Exception as e:
            logger.warning(f"Failed to update cache status (lightweight): {e}")
            self.cache_status_var.set("Cache status error")

    def _show_map_art_toggle_message(self, new_state: bool):
        """Show the map art toggle information message"""
        try:
            if new_state:
                current_model = self.config.clip.model_name
                if current_model == "ViT-L/14":
                    message = ("Map art detection enabled! 🎯\n\n"
                             "✅ Images will be analyzed to detect and crop map artwork before similarity search\n"
                             "⚡ Auto-disabled during cache building for speed\n"
                             "🎯 Usually more accurate than using full images")
                else:
                    message = ("Map art detection enabled! 🎯\n\n"
                             "✅ Images will be analyzed to detect and crop map artwork before similarity search\n"
                             "⚡ Auto-disabled during cache building for speed\n"
                             "🎯 Usually more accurate than using full images")
                messagebox.showinfo("Map Art Detection Enabled", message)
            else:
                messagebox.showinfo("Map Art Detection Disabled", 
                                  "Map art detection disabled. 📋\n\n"
                                  "• Full images will be used for similarity search\n"
                                  "• Faster processing but may be less accurate for map artwork\n"
                                  "• Cache building performance unaffected")
        except Exception as e:
            logger.error(f"Failed to show map art toggle message: {e}")