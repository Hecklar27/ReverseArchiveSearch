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

from core.config import Config
from data.discord_parser import DiscordParser
from data.models import DiscordMessage, SearchResult, ProcessingStats
from search.strategies import SearchEngine

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
        
        ttk.Label(self.control_frame, text="HTML Archive:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.html_file_var = tk.StringVar()
        self.html_file_entry = ttk.Entry(self.control_frame, textvariable=self.html_file_var, width=50)
        self.html_file_entry.grid(row=1, column=1, padx=(5, 0), pady=2, sticky=tk.EW)
        ttk.Button(self.control_frame, text="Browse", 
                  command=self._browse_html_archive).grid(row=1, column=2, padx=(5, 0), pady=2)
        
        # Search Controls
        search_frame = ttk.Frame(self.control_frame)
        search_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=tk.EW)
        
        # Real-time search button
        self.search_button = ttk.Button(search_frame, text="Search (Real-time)", 
                                       command=self._start_search)
        self.search_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cache controls (Phase 2)
        cache_frame = ttk.LabelFrame(search_frame, text="Phase 2: Cache Mode", padding="5")
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
            
            # Create detailed device text
            device_text = f"Device: {device_info['device'].upper()}"
            
            if device_info.get('using_cuda', False):
                # Using CUDA - show GPU info
                gpu_name = device_info.get('gpu_name', 'Unknown GPU')
                gpu_memory = device_info.get('gpu_memory_gb', 'Unknown')
                device_text += f" ({gpu_name}, {gpu_memory}GB)"
            elif device_info.get('gpu_available', False):
                # CUDA available but not used
                gpu_name = device_info.get('gpu_name', 'Unknown GPU')
                device_text += f" (GPU available: {gpu_name} - not used)"
            else:
                # No CUDA support
                device_text += " (No CUDA support)"
            
            self.device_label.config(text=device_text)
            
            # Log detailed info for debugging
            logger.info(f"Device Status: {device_info.get('cuda_status', 'Unknown')}")
            if device_info.get('cuda_error'):
                logger.warning(f"CUDA Error: {device_info['cuda_error']}")
                
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
        self.search_button.config(state='normal')
        
        # Update results display
        self._display_results(results, stats)
        
        # Update progress text
        self._update_progress(f"Search complete - found {len(results)} results in {stats.processing_time_seconds:.1f}s")
    
    def _on_search_error(self, error_message: str):
        """Handle search error"""
        self.progress_bar.grid_remove()
        self.progress_var.set("Search failed")
        self.search_button.config(state='normal')
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
    
    def _start_cache_build(self):
        """Start building the embedding cache"""
        if not self.discord_messages:
            self._show_error("Please load HTML archive file first")
            return
        
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
                success = self.search_engine.build_cache(self.discord_messages, progress_callback)
                self.root.after(0, lambda: self._on_cache_build_complete(success))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._on_cache_build_error(error_msg))
        
        # Start cache building in background thread
        threading.Thread(target=cache_build_worker, daemon=True).start()
    
    def _on_cache_build_complete(self, success: bool):
        """Handle cache build completion"""
        self.progress_bar.grid_remove()
        
        if success:
            self.progress_var.set("Cache built successfully")
            self._update_cache_status()
            messagebox.showinfo("Cache Built", 
                              "Embedding cache built successfully!\nCached searches will now be much faster.")
        else:
            self.progress_var.set("Cache build failed")
            self._show_error("Failed to build cache. Check logs for details.")
        
        # Re-enable buttons
        self.preprocess_button.config(state='normal')
        self.search_button.config(state='normal')
    
    def _on_cache_build_error(self, error_message: str):
        """Handle cache build error"""
        self.progress_bar.grid_remove()
        self.progress_var.set("Cache build failed")
        self.preprocess_button.config(state='normal')
        self.search_button.config(state='normal')
        self._show_error(f"Cache build failed: {error_message}")
        logger.error(f"Cache build error: {error_message}")
    
    def _update_cache_progress(self, progress_percent: float, progress_text: str):
        """Update cache build progress"""
        self.progress_bar.config(value=progress_percent)
        self.progress_var.set(f"Building cache: {progress_text}")
        self.root.update_idletasks()
    
    def _start_cached_search(self):
        """Start cached search"""
        if not self.user_image_path:
            self._show_error("Please select a user image first")
            return
        
        if not self.discord_messages:
            self._show_error("Please load HTML archive file first")
            return
        
        if not self.search_engine.has_cache():
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
        """Clear the embedding cache"""
        if messagebox.askyesno("Clear Cache", 
                              "Are you sure you want to clear the embedding cache?\n"
                              "This will remove all pre-computed embeddings."):
            try:
                self.search_engine.clear_cache()
                self._update_cache_status()
                messagebox.showinfo("Cache Cleared", "Embedding cache cleared successfully.")
                self.progress_var.set("Cache cleared")
            except Exception as e:
                self._show_error(f"Failed to clear cache: {e}")
    
    def _update_cache_status(self):
        """Update cache status display"""
        try:
            if self.search_engine.has_cache():
                cache_stats = self.search_engine.get_cache_stats()
                
                if cache_stats.get('status') == 'valid':
                    total_images = cache_stats.get('total_images', 0)
                    age_days = cache_stats.get('age_days', 0)
                    
                    status_text = f"✓ {total_images} images cached"
                    if age_days > 0:
                        status_text += f" ({age_days}d old)"
                    
                    self.cache_status_var.set(status_text)
                    self.cached_search_button.config(state='normal')
                    self.clear_cache_button.config(state='normal')
                else:
                    # Check if cache is incompatible vs just invalid
                    cache_manager = self.search_engine.cache_manager
                    if cache_manager._load_metadata():  # Cache exists but invalid
                        self.cache_status_var.set("⚠ Incompatible cache - rebuild needed")
                    else:
                        self.cache_status_var.set("⚠ Invalid cache")
                    
                    self.cached_search_button.config(state='disabled')
                    self.clear_cache_button.config(state='normal')  # Allow clearing incompatible cache
            else:
                self.cache_status_var.set("No cache")
                self.cached_search_button.config(state='disabled')
                self.clear_cache_button.config(state='disabled')
                
            # Enable pre-process button if we have Discord data
            if self.discord_messages:
                self.preprocess_button.config(state='normal')
            else:
                self.preprocess_button.config(state='disabled')
                
        except Exception as e:
            logger.warning(f"Failed to update cache status: {e}")
            self.cache_status_var.set("Cache error")
    
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
            
            # First, check if we have any saved parsed messages at all
            if not cache_manager.parsed_messages_file.exists() or not cache_manager.parsed_metadata_file.exists():
                logger.info("No previously parsed messages found")
                return
            
            # Load the metadata to get the HTML file path
            with open(cache_manager.parsed_metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            html_path = metadata.html_file_path
            logger.info(f"Found previously parsed messages for HTML file: {html_path}")
            
            # Check if this HTML file is still valid
            if cache_manager.has_parsed_messages(html_path):
                logger.info(f"Loading previously parsed messages for {html_path}")
                
                # Load the messages
                messages = cache_manager.load_parsed_messages()
                if messages:
                    self.discord_messages = messages
                    self.html_archive_path = Path(html_path)
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