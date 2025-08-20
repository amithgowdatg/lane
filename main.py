import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
import os
import math
import threading
import time

try:
    from lane_detector import LaneDetector
except ImportError:
    print("Warning: Could not import LaneDetector from lane_detector.py")
    class LaneDetector:
        def __init__(self):
            self.left_a, self.left_b, self.left_c = [], [], []
            self.right_a, self.right_b, self.right_c = [], [], []
            self.lane_distance_threshold_m = 4.0
        
        def process_frame(self, frame):
            processed = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_detection = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            cv2.putText(processed, "Fallback Mode - Import lane_detector.py", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return processed, edge_detection

class BasicLaneDetector:
    """Part 1 - Basic Lane Detection with Pipeline Visualization"""
    
    def __init__(self):
        pass
    
    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def canny(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]  
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=10):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def slope_lines(self, image, lines):
        img = image.copy()
        poly_vertices = []
        order = [0,1,3,2] 
        left_lines = []
        right_lines = []

        for line in lines:
            for x1,y1,x2,y2 in line:
                if x1 == x2:
                    pass
                else:
                    m = (y2 - y1) / (x2 - x1)
                    c = y1 - m * x1
                    if m < 0: 
                        left_lines.append((m,c))
                    elif m >= 0: 
                        right_lines.append((m,c))

        if left_lines:
            left_line = np.mean(left_lines, axis=0)
            for slope, intercept in [left_line]:
                rows, cols = image.shape[:2]
                y1= int(rows) 
                y2= int(rows*0.6) 
                x1=int((y1-intercept)/slope)
                x2=int((y2-intercept)/slope)
                poly_vertices.append((x1, y1))
                poly_vertices.append((x2, y2))

        if right_lines:
            right_line = np.mean(right_lines, axis=0)
            for slope, intercept in [right_line]:
                rows, cols = image.shape[:2]
                y1= int(rows) 
                y2= int(rows*0.6) 
                x1=int((y1-intercept)/slope)
                x2=int((y2-intercept)/slope)
                poly_vertices.append((x1, y1))
                poly_vertices.append((x2, y2))

        if len(poly_vertices) == 4:
            poly_vertices = [poly_vertices[i] for i in order]
            cv2.fillPoly(img, pts = np.array([poly_vertices],'int32'), color = (0,255,0))

        return cv2.addWeighted(image,0.7,img,0.4,0.)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if lines is not None:
            line_img = self.slope_lines(line_img, lines)
        return line_img

    def weighted_img(self, img, initial_img, α=0.1, β=1., γ=0.):
        img = img.astype(np.uint8)
        initial_img = initial_img.astype(np.uint8)
        lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
        return lines_edges

    def get_vertices(self, image):
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.15, rows]
        top_left     = [cols*0.45, rows*0.6]
        bottom_right = [cols*0.95, rows]
        top_right    = [cols*0.55, rows*0.6]
        ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return ver

    def lane_finding_pipeline(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        input_img = image.copy()
        gray_img = self.grayscale(image)
        smoothed_img = self.gaussian_blur(img = gray_img, kernel_size = 5)
        smoothed_img_uint8 = smoothed_img.astype(np.uint8)
        canny_img = self.canny(img = smoothed_img_uint8, low_threshold = 180, high_threshold = 240)
        masked_img = self.region_of_interest(img = canny_img, vertices = self.get_vertices(image))
        houghed_lines = self.hough_lines(img = masked_img, rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 20, max_line_gap = 180)
        output = self.weighted_img(img = houghed_lines, initial_img = image, α=0.8, β=1., γ=0.)

        return input_img, gray_img, smoothed_img, canny_img, masked_img, houghed_lines, output

class ModernLaneDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ADAS - Lane Detection and Driver Alertness System")
        self.root.configure(bg='#0f0f0f')
        self.root.geometry("1600x1000")
        self.root.state('zoomed')  
        
        self.setup_styles()
        
        self.advanced_detector = LaneDetector()  
        self.basic_detector = BasicLaneDetector()  
        self.video_source = None
        self.current_frame = None
        self.video_playing = False
        self.detection_method = tk.StringVar(value="Advanced")
        self.processing_thread = None
        self.progress_var = tk.DoubleVar()
        self.fps_var = tk.StringVar(value="0 FPS")
        self.frame_count = 0
        self.start_time = time.time()
        self.video_fps = 30  
        self.video_frame_delay = 33  
        self.video_thread_running = False
        
        
        self.create_widgets()
        
        
        self.update_fps()
        
    def setup_styles(self):
        """Configure modern dark theme styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        
        self.colors = {
            'bg_primary': "#203234",
            'bg_secondary': '#1a1a1a',
            'bg_tertiary': '#2d2d2d',
            'accent': '#00ff41',
            'accent_hover': '#00cc33',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'warning': '#ff6b35',
            'error': '#ff3333',
            'success': '#00ff41'
        }
        
    
        self.style.configure('TFrame', background=self.colors['bg_primary'])
        self.style.configure('Card.TFrame', background=self.colors['bg_secondary'], relief='flat', borderwidth=1)
        
        self.style.configure('Title.TLabel',
                            background=self.colors['bg_primary'],
                            foreground=self.colors['accent'],
                            font=('Segoe UI', 28, 'bold'))
        
        self.style.configure('Subtitle.TLabel',
                            background=self.colors['bg_primary'],
                            foreground=self.colors['text_secondary'],
                            font=('Segoe UI', 12))
        
        self.style.configure('Modern.TLabel',
                            background=self.colors['bg_secondary'],
                            foreground=self.colors['text_primary'],
                            font=('Segoe UI', 11))
        
        self.style.configure('Status.TLabel',
                            background=self.colors['bg_primary'],
                            foreground=self.colors['text_secondary'],
                            font=('Segoe UI', 10))
        
       
        self.style.configure('Primary.TButton',
                            background=self.colors['accent'],
                            foreground='#000000',
                            borderwidth=0,
                            focuscolor='none',
                            padding=(20, 10),
                            font=('Segoe UI', 10, 'bold'))
        
        self.style.configure('Secondary.TButton',
                            background=self.colors['bg_tertiary'],
                            foreground=self.colors['text_primary'],
                            borderwidth=0,
                            focuscolor='none',
                            padding=(20, 10),
                            font=('Segoe UI', 10, 'bold'))
        
        self.style.configure('Danger.TButton',
                            background=self.colors['error'],
                            foreground='#ffffff',
                            borderwidth=0,
                            focuscolor='none',
                            padding=(20, 10),
                            font=('Segoe UI', 10, 'bold'))
    
        self.style.configure('Modern.Horizontal.TProgressbar',
                            background=self.colors['accent'],
                            troughcolor=self.colors['bg_tertiary'],
                            borderwidth=0,
                            lightcolor=self.colors['accent'],
                            darkcolor=self.colors['accent'])
        
       
        self.style.configure('Modern.TNotebook',
                            background=self.colors['bg_primary'],
                            borderwidth=0,
                            tabmargins=[0, 0, 0, 0])
        
        self.style.configure('Modern.TNotebook.Tab',
                            background=self.colors['bg_tertiary'],
                            foreground=self.colors['text_primary'],
                            padding=(20, 10),
                            font=('Segoe UI', 10, 'bold'))
        
        self.style.map('Modern.TNotebook.Tab',
                      background=[('selected', self.colors['accent'])],
                      foreground=[('selected', '#000000')])
        
    def create_widgets(self):
        
        main_container = ttk.Frame(self.root, padding="20")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(2, weight=1)
        

        self.create_header(main_container)
        
       
        self.create_control_panel(main_container)
        
        self.create_main_content(main_container)
        
       
        self.create_status_bar(main_container)
        
    def create_header(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        header_frame.columnconfigure(1, weight=1)
        
        title_label = ttk.Label(header_frame, 
                               text="ADAS - Lane Detection System",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        subtitle_label = ttk.Label(header_frame,
                                  text="Advanced Driver Assistance System with Real-time Processing",
                                  style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, sticky=tk.W)
        
        
        fps_label = ttk.Label(header_frame, textvariable=self.fps_var,
                             style='Status.TLabel')
        fps_label.grid(row=0, column=1, sticky=tk.E)
        
    def create_control_panel(self, parent):
        control_frame = ttk.Frame(parent, style='Card.TFrame', padding="20")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        control_frame.columnconfigure(1, weight=1)
        
       
        method_frame = ttk.Frame(control_frame)
        method_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(method_frame, text="Detection Method:",
                  style='Modern.TLabel').pack(side=tk.LEFT, padx=(0, 15))
       
        self.create_radio_buttons(method_frame)
        
      
        self.create_action_buttons(control_frame)
        
        self.progress_bar = ttk.Progressbar(control_frame,
                                           style='Modern.Horizontal.TProgressbar',
                                           variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(15, 0))
        
    def create_radio_buttons(self, parent):
        radio_frame = ttk.Frame(parent)
        radio_frame.pack(side=tk.LEFT)
        
       
        advanced_radio = ttk.Radiobutton(radio_frame,
                                        text="lane departure",
                                        variable=self.detection_method,
                                        value="Advanced")
        advanced_radio.pack(side=tk.LEFT, padx=(0, 20))
        
      
        basic_radio = ttk.Radiobutton(radio_frame,
                                     text="lane detection",
                                     variable=self.detection_method,
                                     value="Basic")
        basic_radio.pack(side=tk.LEFT, padx=(0, 20))
        
    def create_action_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(0, 15))
        
       
        ttk.Button(button_frame, text="📁 Upload Video",
                   style='Primary.TButton',
                   command=self.select_video).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="🖼️ Upload Image",
                   style='Primary.TButton',
                   command=self.select_image).pack(side=tk.LEFT, padx=(0, 10))
        
       
        ttk.Button(button_frame, text="⚡ Process Frame",
                   style='Secondary.TButton',
                   command=self.process_current_frame).pack(side=tk.LEFT, padx=(0, 10))
        
        self.play_btn = ttk.Button(button_frame, text="▶️ Play",
                                  style='Secondary.TButton',
                                  command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="🔍 Show Pipeline",
                   style='Secondary.TButton',
                   command=self.show_pipeline).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="🔄 Reset",
                   style='Danger.TButton',
                   command=self.reset_system).pack(side=tk.RIGHT)
        
    def create_main_content(self, parent):
       
        self.notebook = ttk.Notebook(parent, style='Modern.TNotebook')
        self.notebook.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
       
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="🎯 Main Detection")
        
       
        self.video_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.video_tab, text="🎬 Video Processing")
        
       
        self.pipeline_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_tab, text="⚙️ Pipeline Visualization")
        
       
        self.setup_main_tab()
        self.setup_video_tab()
        self.setup_pipeline_tab()
        
    def setup_main_tab(self):
        
        display_container = ttk.Frame(self.main_tab, padding="20")
        display_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    
        self.main_tab.columnconfigure(0, weight=1)
        self.main_tab.rowconfigure(0, weight=1)
        display_container.columnconfigure((0, 1, 2), weight=1)
        display_container.rowconfigure(0, weight=1)
        
        self.create_display_card(display_container, "Original Input", 0, 0)
        self.create_display_card(display_container, "Lane Detection", 0, 1)
        self.create_display_card(display_container, "Edge Processing", 0, 2)
        
    def create_display_card(self, parent, title, row, col):
        
        card_frame = ttk.Frame(parent, style='Card.TFrame', padding="15")
        card_frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10)
        card_frame.columnconfigure(0, weight=1)
        card_frame.rowconfigure(1, weight=1)
        
       
        title_label = ttk.Label(card_frame, text=title, style='Modern.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 10))
        
      
        canvas = tk.Canvas(card_frame, 
                          width=450, height=300,
                          bg=self.colors['bg_primary'],
                          highlightthickness=2,
                          highlightbackground=self.colors['accent'])
        canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        if col == 0:
            self.original_canvas = canvas
        elif col == 1:
            self.processed_canvas = canvas
        else:
            self.edge_canvas = canvas
        
        canvas.create_text(225, 150, text=f"No {title.lower()} loaded",
                          fill=self.colors['text_secondary'],
                          font=('Segoe UI', 12))
        
    def setup_video_tab(self):
        video_container = ttk.Frame(self.video_tab, padding="20")
        video_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
       
        self.video_tab.columnconfigure(0, weight=1)
        self.video_tab.rowconfigure(0, weight=1)
        video_container.columnconfigure((0, 1), weight=1)
        video_container.rowconfigure(1, weight=1)
        
        controls_frame = ttk.Frame(video_container, style='Card.TFrame', padding="20")
        controls_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        ttk.Label(controls_frame, text="Video Processing Controls - Real-time Lane Detection",
                  style='Modern.TLabel').pack(anchor=tk.W)
        
      
        self.create_video_display_card(video_container, "Original Video", 1, 0)
        self.create_video_display_card(video_container, "Processed Video", 1, 1)
        
    def create_video_display_card(self, parent, title, row, col):
       
        card_frame = ttk.Frame(parent, style='Card.TFrame', padding="15")
        card_frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10)
        card_frame.columnconfigure(0, weight=1)
        card_frame.rowconfigure(1, weight=1)
        
       
        title_label = ttk.Label(card_frame, text=title, style='Modern.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 10))
        
      
        canvas = tk.Canvas(card_frame,
                          width=600, height=400,
                          bg=self.colors['bg_primary'],
                          highlightthickness=2,
                          highlightbackground=self.colors['accent'])
        canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    
        if col == 0:
            self.video_original_canvas = canvas
        else:
            self.video_processed_canvas = canvas
         
        canvas.create_text(300, 200, text=f"No {title.lower()} loaded",
                          fill=self.colors['text_secondary'],
                          font=('Segoe UI', 12))
        
    def setup_pipeline_tab(self):
        self.pipeline_frame = ttk.Frame(self.pipeline_tab, padding="20")
        self.pipeline_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
       
        self.pipeline_tab.columnconfigure(0, weight=1)
        self.pipeline_tab.rowconfigure(0, weight=1)
        self.pipeline_frame.columnconfigure(0, weight=1)
        self.pipeline_frame.rowconfigure(1, weight=1)
        
       
        ttk.Label(self.pipeline_frame, text="Processing Pipeline Visualization",
                  style='Modern.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        
        
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.patch.set_facecolor(self.colors['bg_primary'])
        self.fig.suptitle('Lane Detection Pipeline', color=self.colors['accent'], fontsize=16, fontweight='bold')
        
        
        for i in range(2):
            for j in range(3):
                if i < len(self.axes) and j < len(self.axes[i]):
                    self.axes[i][j].set_xticks([])
                    self.axes[i][j].set_yticks([])
                    self.axes[i][j].set_facecolor(self.colors['bg_secondary'])
        
        self.canvas_widget = FigureCanvasTkAgg(self.fig, self.pipeline_frame)
        self.canvas_widget.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def create_status_bar(self, parent):
        status_frame = ttk.Frame(parent, style='Card.TFrame', padding="15")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        status_frame.columnconfigure(1, weight=1)
        
       
        self.status_indicator = tk.Canvas(status_frame, width=20, height=20,
                                         bg=self.colors['bg_secondary'],
                                         highlightthickness=0)
        self.status_indicator.grid(row=0, column=0, padx=(0, 10))
        self.status_indicator.create_oval(5, 5, 15, 15, fill=self.colors['success'], outline="")
        
        
        self.status_var = tk.StringVar(value="System ready - Select detection method and load media")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                style='Status.TLabel')
        status_label.grid(row=0, column=1, sticky=tk.W)
        
       
        info_label = ttk.Label(status_frame, text="ADAS v2.0 | OpenCV Ready | GPU Acceleration: Available",
                              style='Status.TLabel')
        info_label.grid(row=0, column=2, sticky=tk.E)
        
    def update_status(self, message, status_type="info"):
        """Update status bar with colored indicator"""
        self.status_var.set(message)
        
      
        self.status_indicator.delete("all")
        
       
        color_map = {
            "info": self.colors['accent'],
            "success": self.colors['success'],
            "warning": self.colors['warning'],
            "error": self.colors['error']
        }
        
        color = color_map.get(status_type, self.colors['accent'])
        self.status_indicator.create_oval(5, 5, 15, 15, fill=color, outline="")
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        if current_time - self.start_time >= 1.0:
            fps = self.frame_count / (current_time - self.start_time)
            self.fps_var.set(f"{fps:.1f} FPS")
            self.frame_count = 0
            self.start_time = current_time
        
        self.root.after(100, self.update_fps)
        
    def display_frame(self, canvas, frame, target_width=450, target_height=300):
        """Display frame on canvas with proper scaling"""
        if frame is None:
            return
            
        try:
           
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
                
            height, width = frame_rgb.shape[:2]
            aspect_ratio = width / height
            
            if aspect_ratio > target_width / target_height:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            resized_frame = cv2.resize(frame_rgb, (new_width, new_height))
            
            pil_image = Image.fromarray(resized_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            
            canvas.delete("all")
            canvas.create_image(target_width//2, target_height//2, image=photo, anchor='center')
            
            canvas.image = photo
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
            canvas.delete("all")
            canvas.create_text(target_width//2, target_height//2, 
                             text="Error displaying frame",
                             fill=self.colors['error'],
                             font=('Segoe UI', 12))

    def select_video(self):
        """Select video file for processing"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_source = file_path
            self.load_video()
            self.update_status(f"Video loaded: {os.path.basename(file_path)}", "success")
            self.notebook.select(self.video_tab)

    def select_image(self):
        """Select image file for processing"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
            self.update_status(f"Image loaded: {os.path.basename(file_path)}", "success")
            self.notebook.select(self.main_tab)

    def load_video(self):
        """Load video file and prepare for processing"""
        if not self.video_source:
            return
            
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise ValueError("Could not open video file")
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_frame_delay = int(1000 / self.video_fps) if self.video_fps > 0 else 33
            
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(self.video_original_canvas, frame, 600, 400)
                
                if self.detection_method.get() == "Advanced":
                    processed_frame, edge_frame = self.advanced_detector.process_frame(frame)
                else:
                    processed_frame = self.process_basic_detection(frame)
                    
                self.display_frame(self.video_processed_canvas, processed_frame, 600, 400)
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        except Exception as e:
            self.update_status(f"Error loading video: {str(e)}", "error")
            messagebox.showerror("Error", f"Could not load video: {str(e)}")

    def load_image(self, file_path):
        """Load image file for processing"""
        try:
            
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not load image file")
                
            self.current_frame = image
            
            self.display_frame(self.original_canvas, image)
            
            
            self.process_current_frame()
            
        except Exception as e:
            self.update_status(f"Error loading image: {str(e)}", "error")
            messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def process_current_frame(self):
        """Process current frame with selected detection method"""
        if self.current_frame is None:
            self.update_status("No frame to process", "warning")
            return
            
        try:
            if self.detection_method.get() == "Advanced":
                
                processed_frame, edge_frame = self.advanced_detector.process_frame(self.current_frame)
                
                
                self.display_frame(self.processed_canvas, processed_frame)
                self.display_frame(self.edge_canvas, edge_frame)
                
            else:
                processed_frame = self.process_basic_detection(self.current_frame)
                
                self.display_frame(self.processed_canvas, processed_frame)
                
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                self.display_frame(self.edge_canvas, edge_colored)
                
            self.update_status("Frame processed successfully", "success")
            
        except Exception as e:
            self.update_status(f"Error processing frame: {str(e)}", "error")
            print(f"Processing error: {e}")

    def process_basic_detection(self, frame):
        """Process frame using basic detection method"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.basic_detector.lane_finding_pipeline(frame_rgb)
            
            output = results[-1]
            
        
            if len(output.shape) == 3:
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            else:
                output_bgr = output
                
            return output_bgr
            
        except Exception as e:
            print(f"Basic detection error: {e}")
            return frame

    def toggle_play(self):
        """Toggle video playback"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.update_status("No video loaded", "warning")
            return
            
        if self.video_playing:
            self.stop_video()
        else:
            self.play_video()

    def play_video(self):
        """Start video playback with processing"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return
            
        self.video_playing = True
        self.video_thread_running = True
        self.play_btn.configure(text="⏸️ Pause")
        self.update_status("Playing video with real-time processing", "info")
        
        self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
        self.video_thread.start()

    def stop_video(self):
        """Stop video playback"""
        self.video_playing = False
        self.video_thread_running = False
        self.play_btn.configure(text="▶️ Play")
        self.update_status("Video paused", "info")

    def video_processing_loop(self):
        """Main video processing loop - runs in separate thread"""
        frame_count = 0
        
        while self.video_thread_running and hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            try:
                if self.detection_method.get() == "Advanced":
                    processed_frame, _ = self.advanced_detector.process_frame(frame)
                else:
                    processed_frame = self.process_basic_detection(frame)
                
                self.root.after(0, self.update_video_display, frame, processed_frame)
                
                progress = (frame_count / self.total_frames) * 100
                self.root.after(0, self.update_progress, progress)
                
                frame_count += 1
                self.frame_count += 1
                
                time.sleep(self.video_frame_delay / 1000.0)
                
            except Exception as e:
                print(f"Video processing error: {e}")
                continue
                
        self.root.after(0, self.video_playback_finished)

    def update_video_display(self, original_frame, processed_frame):
        """Update video display canvases - called from main thread"""
        try:
            self.display_frame(self.video_original_canvas, original_frame, 600, 400)
            self.display_frame(self.video_processed_canvas, processed_frame, 600, 400)
        except Exception as e:
            print(f"Display update error: {e}")

    def video_playback_finished(self):
        """Handle video playback completion"""
        self.video_playing = False
        self.video_thread_running = False
        self.play_btn.configure(text="▶️ Play")
        self.update_status("Video playback completed", "success")
        self.update_progress(0)

    def show_pipeline(self):
        """Show detailed pipeline visualization"""
        if self.current_frame is None:
            self.update_status("No frame to analyze", "warning")
            return
            
        try:
            self.notebook.select(self.pipeline_tab)
            
            if self.detection_method.get() == "Basic":
                self.show_basic_pipeline()
            else:
                self.show_advanced_pipeline()
                
        except Exception as e:
            self.update_status(f"Error showing pipeline: {str(e)}", "error")

    def show_basic_pipeline(self):
        """Show basic detection pipeline steps"""
        try:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            results = self.basic_detector.lane_finding_pipeline(frame_rgb)
            input_img, gray_img, smoothed_img, canny_img, masked_img, houghed_lines, output = results
            
            for i in range(2):
                for j in range(3):
                    if i < len(self.axes) and j < len(self.axes[i]):
                        self.axes[i][j].clear()
                        self.axes[i][j].set_xticks([])
                        self.axes[i][j].set_yticks([])
            
            steps = [
                (input_img, "1. Original Image"),
                (gray_img, "2. Grayscale"),
                (smoothed_img, "3. Gaussian Blur"),
                (canny_img, "4. Canny Edge Detection"),
                (masked_img, "5. Region of Interest"),
                (output, "6. Final Output")
            ]
            
            for idx, (img, title) in enumerate(steps):
                row = idx // 3
                col = idx % 3
                
                if row < len(self.axes) and col < len(self.axes[row]):
                    if len(img.shape) == 2:
                        self.axes[row][col].imshow(img, cmap='gray')
                    else:
                        self.axes[row][col].imshow(img)
                    
                    self.axes[row][col].set_title(title, color='white', fontsize=10)
                    self.axes[row][col].set_xticks([])
                    self.axes[row][col].set_yticks([])
            
            self.canvas_widget.draw()
            self.update_status("Basic pipeline visualization updated", "success")
            
        except Exception as e:
            self.update_status(f"Error in basic pipeline: {str(e)}", "error")
            print(f"Basic pipeline error: {e}")

    def show_advanced_pipeline(self):
        """Show advanced detection pipeline steps"""
        try:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            binary_img = self.advanced_detector.pipeline(frame_rgb)
            warped_img, M = self.advanced_detector.perspective_warp(binary_img)
            
            out_img, curves, lanes, ploty = self.advanced_detector.sliding_window(warped_img, draw_windows=True)
            
            processed_frame, edge_frame = self.advanced_detector.process_frame(self.current_frame)
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            edge_rgb = cv2.cvtColor(edge_frame, cv2.COLOR_BGR2RGB)
            
            for i in range(2):
                for j in range(3):
                    if i < len(self.axes) and j < len(self.axes[i]):
                        self.axes[i][j].clear()
                        self.axes[i][j].set_xticks([])
                        self.axes[i][j].set_yticks([])
            
            steps = [
                (frame_rgb, "1. Original Image"),
                (binary_img, "2. Binary Threshold"),
                (warped_img, "3. Perspective Transform"),
                (out_img, "4. Sliding Window"),
                (edge_rgb, "5. Edge Processing"),
                (processed_rgb, "6. Final Output")
            ]
            
            for idx, (img, title) in enumerate(steps):
                row = idx // 3
                col = idx % 3
                
                if row < len(self.axes) and col < len(self.axes[row]):
                    if len(img.shape) == 2:
                        self.axes[row][col].imshow(img, cmap='gray')
                    else:
                        self.axes[row][col].imshow(img)
                    
                    self.axes[row][col].set_title(title, color='white', fontsize=10)
                    self.axes[row][col].set_xticks([])
                    self.axes[row][col].set_yticks([])
            
            self.canvas_widget.draw()
            self.update_status("Advanced pipeline visualization updated", "success")
            
        except Exception as e:
            self.update_status(f"Error in advanced pipeline: {str(e)}", "error")
            print(f"Advanced pipeline error: {e}")

    def reset_system(self):
        """Reset the entire system"""
        try:
            if hasattr(self, 'video_playing') and self.video_playing:
                self.stop_video()
            
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            self.video_source = None
            self.current_frame = None
            self.video_playing = False
            self.video_thread_running = False
            
            self.clear_all_displays()
            
            self.update_progress(0)
            
            self.detection_method.set("Advanced")
            
            self.advanced_detector = LaneDetector()
            self.basic_detector = BasicLaneDetector()
            
            self.update_status("System reset successfully", "success")
            
        except Exception as e:
            self.update_status(f"Error resetting system: {str(e)}", "error")

    def clear_all_displays(self):
        """Clear all display canvases"""
        canvases = [
            (self.original_canvas, "Original Input", 450, 300),
            (self.processed_canvas, "Lane Detection", 450, 300),
            (self.edge_canvas, "Edge Processing", 450, 300),
            (self.video_original_canvas, "Original Video", 600, 400),
            (self.video_processed_canvas, "Processed Video", 600, 400)
        ]
        
        for canvas, text, width, height in canvases:
            canvas.delete("all")
            canvas.create_text(width//2, height//2, 
                             text=f"No {text.lower()} loaded",
                             fill=self.colors['text_secondary'],
                             font=('Segoe UI', 12))
        
        for i in range(2):
            for j in range(3):
                if i < len(self.axes) and j < len(self.axes[i]):
                    self.axes[i][j].clear()
                    self.axes[i][j].set_xticks([])
                    self.axes[i][j].set_yticks([])
                    self.axes[i][j].set_facecolor(self.colors['bg_secondary'])
        
        self.canvas_widget.draw()

    def on_closing(self):
        """Handle application closing"""
        try:
            if hasattr(self, 'video_playing') and self.video_playing:
                self.stop_video()
            
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error closing application: {e}")
            self.root.destroy()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ModernLaneDetectionGUI(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()