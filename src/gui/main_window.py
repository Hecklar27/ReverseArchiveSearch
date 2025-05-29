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

from ..core.config import Config
from ..data.discord_parser import DiscordParser
from ..data.models import DiscordMessage, SearchResult, ProcessingStats
from ..search.strategies import SearchEngine

logger = logging.getLogger(__name__)

class MainWindow:
    """Main application window"""
    
    def __init__(self, root: tk.Tk, config: Config):
        self.root = root
        self.config = config
        self.discord_parser = DiscordParser()
        self.search_engine = SearchEngine(config)
        
        # State variables
        self.discord_messages: List[DiscordMessage] = []
        self.current_results: List[SearchResult] = []
        self.user_image_path: Optional[Path] = None
        self.discord_json_path: Optional[Path] = None
        
        self._setup_window()
        self._create_widgets()
        self._setup_layout()
        
        # Show device info on startup
        self._show_device_info()
    
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
        
        ttk.Label(self.control_frame, text="Discord JSON:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.json_file_var = tk.StringVar()
        self.json_file_entry = ttk.Entry(self.control_frame, textvariable=self.json_file_var, width=50)
        self.json_file_entry.grid(row=1, column=1, padx=(5, 0), pady=2, sticky=tk.EW)
        ttk.Button(self.control_frame, text="Browse", 
                  command=self._browse_discord_json).grid(row=1, column=2, padx=(5, 0), pady=2)
        
        # Search Controls
        search_frame = ttk.Frame(self.control_frame)
        search_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=tk.EW)
        
        self.search_button = ttk.Button(search_frame, text="Search (Real-time)", 
                                       command=self._start_search)
        self.search_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cache option (Phase 2)
        self.use_cache_var = tk.BooleanVar()
        self.cache_checkbox = ttk.Checkbutton(search_frame, text="Use Cache (Phase 2)", 
                                            variable=self.use_cache_var, state='disabled')
        self.cache_checkbox.pack(side=tk.LEFT, padx=(0, 10))
        
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
        
        self.details_text = tk.Text(details_frame, height=4, wrap=tk.WORD, state='disabled')
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Discord link button
        self.discord_link_button = ttk.Button(details_frame, text="Open in Discord", 
                                            command=self._open_discord_link, state='disabled')
        self.discord_link_button.pack(side=tk.BOTTOM, pady=(10, 0))
        
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
            device_info = self.search_engine.get_device_info()
            device_text = f"Device: {device_info['device'].upper()}"
            if device_info['device'] == 'cuda':
                device_text += f" ({device_info.get('gpu_name', 'Unknown GPU')})"
            self.device_label.config(text=device_text)
        except Exception as e:
            self.device_label.config(text="Device: Error")
            logger.warning(f"Failed to get device info: {e}")
    
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
    
    def _browse_discord_json(self):
        """Open file dialog to select Discord JSON file"""
        filetypes = [
            ('JSON files', '*.json'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Discord JSON Export",
            filetypes=filetypes
        )
        
        if filename:
            self.discord_json_path = Path(filename)
            self.json_file_var.set(str(self.discord_json_path))
            logger.info(f"Selected Discord JSON: {self.discord_json_path}")
            
            # Parse Discord JSON in background
            self._parse_discord_json()
    
    def _parse_discord_json(self):
        """Parse Discord JSON file in background"""
        if not self.discord_json_path:
            return
            
        def parse_worker():
            try:
                self._update_progress("Parsing Discord JSON...")
                self.discord_messages = self.discord_parser.parse_file(self.discord_json_path)
                
                # Update UI in main thread
                self.root.after(0, self._on_discord_parsed)
                
            except Exception as e:
                logger.error(f"Failed to parse Discord JSON: {e}")
                self.root.after(0, lambda: self._show_error(f"Failed to parse Discord JSON:\n{str(e)}"))
        
        threading.Thread(target=parse_worker, daemon=True).start()
    
    def _on_discord_parsed(self):
        """Called when Discord JSON parsing is complete"""
        self._update_progress(f"Loaded {len(self.discord_messages)} messages with images")
        logger.info(f"Parsed {len(self.discord_messages)} Discord messages with images")
    
    def _start_search(self):
        """Start the image search process"""
        # Validate inputs
        if not self.user_image_path or not self.user_image_path.exists():
            self._show_error("Please select a valid user image")
            return
            
        if not self.discord_messages:
            self._show_error("Please select and load a Discord JSON file")
            return
        
        # Check image format
        if not self.config.is_supported_image(self.user_image_path):
            self._show_error(f"Unsupported image format. Supported: {', '.join(self.config.get_supported_image_extensions())}")
            return
        
        # Disable search button and show progress
        self.search_button.config(state='disabled')
        self.progress_bar.grid()
        self.progress_bar['value'] = 0
        
        # Clear previous results
        self._clear_results()
        
        # Start search in background thread
        use_cache = self.use_cache_var.get()
        
        def progress_callback(current: int, total: int, status: str):
            """Progress callback for search updates"""
            progress_percent = (current / total * 100) if total > 0 else 0
            progress_text = f"{status} ({current}/{total} images)"
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_search_progress(progress_percent, progress_text))
        
        def search_worker():
            try:
                self._update_progress("Starting search...")
                
                results, stats = self.search_engine.search(
                    self.user_image_path, 
                    self.discord_messages, 
                    use_cache=use_cache,
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
        self.search_button.config(state='normal')
        
        # Update results display
        self._display_results(results, stats)
        
        # Update progress text
        self._update_progress(f"Search complete - found {len(results)} results in {stats.processing_time_seconds:.1f}s")
    
    def _on_search_error(self, error_message: str):
        """Called when search encounters an error"""
        self.progress_bar.grid_remove()
        self.search_button.config(state='normal')
        
        self._update_progress("Search failed")
        self._show_error(f"Search failed:\n{error_message}")
    
    def _update_search_progress(self, progress_percent: float, progress_text: str):
        """Update search progress bar and text"""
        self.progress_bar['value'] = progress_percent
        self.progress_var.set(progress_text)
    
    def _display_results(self, results: List[SearchResult], stats: ProcessingStats):
        """Display search results in the tree view"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add results
        for i, result in enumerate(results):
            author_name = result.message.author.nickname or result.message.author.name
            similarity_pct = f"{result.similarity_score * 100:.1f}%"
            
            # Format timestamp
            timestamp = result.message.timestamp[:10] if result.message.timestamp else "Unknown"
            
            self.results_tree.insert('', 'end', values=(
                similarity_pct,
                author_name,
                result.attachment.filename,
                timestamp
            ))
        
        # Update results count
        self.results_count_label.config(text=f"{len(results)} results")
        
        # Update statistics in progress
        success_rate = stats.get_success_rate()
        self._update_progress(
            f"Processed {stats.processed_images}/{stats.total_images} images "
            f"({success_rate:.1f}% success) in {stats.processing_time_seconds:.1f}s"
        )
    
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
        details += f"Similarity: {result.similarity_score * 100:.1f}%\n"
        details += f"Timestamp: {result.message.timestamp}\n"
        details += f"Content: {result.message.content[:200]}{'...' if len(result.message.content) > 200 else ''}\n"
        details += f"Discord URL: {result.discord_url}"
        
        self.details_text.insert(1.0, details)
        
        # Disable text widget
        self.details_text.config(state='disabled')
        
        # Enable Discord link button
        self.discord_link_button.config(state='normal')
    
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
        
        self.current_results = []
    
    def _update_progress(self, message: str):
        """Update progress message"""
        self.progress_var.set(message)
        logger.info(message)
    
    def _show_error(self, message: str):
        """Show error dialog"""
        messagebox.showerror("Error", message)
        logger.error(message) 