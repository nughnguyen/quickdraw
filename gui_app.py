"""
QuickDraw GUI Application
Modern interface for drawing recognition with camera controls
Author: nughnguyen
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import *
from src.utils import get_images, get_overlay


class QuickDrawGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickDraw - AI Drawing Recognition")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)
        
        # Application state
        self.is_running = False
        self.is_drawing = False
        self.is_shown = False
        self.camera = None
        self.model = None
        self.predicted_class = None
        
        # Drawing state
        self.points = deque(maxlen=512)
        self.canvas_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Settings
        self.color_choice = tk.StringVar(value="green")
        self.min_area = tk.IntVar(value=3000)
        self.show_canvas = tk.BooleanVar(value=False)
        
        # Color settings
        self.update_color_settings()
        
        # Load model
        self.load_model()
        
        # Load class images
        self.class_images = get_images("images", CLASSES)
        
        # Create UI
        self.create_ui()
        
        # Start update loop
        self.update_frame()
    
    def load_model(self):
        """Load the trained QuickDraw model"""
        try:
            if torch.cuda.is_available():
                self.model = torch.load("trained_models/whole_model_quickdraw")
            else:
                self.model = torch.load("trained_models/whole_model_quickdraw", 
                                       map_location=lambda storage, loc: storage, 
                                       weights_only=False)
            self.model.eval()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def update_color_settings(self):
        """Update color settings based on user choice"""
        color = self.color_choice.get()
        if color == "red":
            self.color_lower = np.array(RED_HSV_LOWER)
            self.color_upper = np.array(RED_HSV_UPPER)
            self.color_pointer = RED_RGB
        elif color == "green":
            self.color_lower = np.array(GREEN_HSV_LOWER)
            self.color_upper = np.array(GREEN_HSV_UPPER)
            self.color_pointer = GREEN_RGB
        else:  # blue
            self.color_lower = np.array(BLUE_HSV_LOWER)
            self.color_upper = np.array(BLUE_HSV_UPPER)
            self.color_pointer = BLUE_RGB
    
    def create_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera feed
        left_panel = tk.Frame(main_frame, bg="#2d2d2d", relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Camera title
        camera_title = tk.Label(left_panel, text="📷 Camera Feed", 
                               font=("Arial", 14, "bold"), bg="#2d2d2d", fg="white")
        camera_title.pack(pady=10)
        
        # Camera display
        self.camera_label = tk.Label(left_panel, bg="black")
        self.camera_label.pack(padx=10, pady=10)
        
        # Canvas display (optional)
        self.canvas_label = tk.Label(left_panel, bg="black")
        
        # Right panel - Controls and info
        right_panel = tk.Frame(main_frame, bg="#2d2d2d", width=350, 
                              relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Title
        title_label = tk.Label(right_panel, text="QuickDraw AI", 
                              font=("Arial", 18, "bold"), bg="#2d2d2d", fg="#4CAF50")
        title_label.pack(pady=15)
        
        # Status indicator
        status_frame = tk.Frame(right_panel, bg="#2d2d2d")
        status_frame.pack(pady=10)
        
        tk.Label(status_frame, text="Status:", font=("Arial", 10), 
                bg="#2d2d2d", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.status_indicator = tk.Label(status_frame, text="●", 
                                        font=("Arial", 16), bg="#2d2d2d", fg="red")
        self.status_indicator.pack(side=tk.LEFT)
        
        self.status_text = tk.Label(status_frame, text="Stopped", 
                                   font=("Arial", 10, "bold"), bg="#2d2d2d", fg="white")
        self.status_text.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        controls_frame = tk.Frame(right_panel, bg="#2d2d2d")
        controls_frame.pack(pady=15, padx=20, fill=tk.X)
        
        self.start_btn = tk.Button(controls_frame, text="▶ Start Camera", 
                                   command=self.start_camera,
                                   font=("Arial", 11, "bold"), bg="#4CAF50", fg="white",
                                   relief=tk.RAISED, borderwidth=3, cursor="hand2",
                                   padx=20, pady=10)
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = tk.Button(controls_frame, text="⏹ Stop Camera", 
                                 command=self.stop_camera,
                                 font=("Arial", 11, "bold"), bg="#f44336", fg="white",
                                 relief=tk.RAISED, borderwidth=3, cursor="hand2",
                                 padx=20, pady=10, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        self.draw_btn = tk.Button(controls_frame, text="✏ Start Drawing", 
                                 command=self.toggle_drawing,
                                 font=("Arial", 11, "bold"), bg="#2196F3", fg="white",
                                 relief=tk.RAISED, borderwidth=3, cursor="hand2",
                                 padx=20, pady=10, state=tk.DISABLED)
        self.draw_btn.pack(fill=tk.X, pady=5)
        
        self.clear_btn = tk.Button(controls_frame, text="🗑 Clear Canvas", 
                                  command=self.clear_canvas,
                                  font=("Arial", 11, "bold"), bg="#FF9800", fg="white",
                                  relief=tk.RAISED, borderwidth=3, cursor="hand2",
                                  padx=20, pady=10, state=tk.DISABLED)
        self.clear_btn.pack(fill=tk.X, pady=5)
        
        # Separator
        ttk.Separator(right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Settings
        settings_label = tk.Label(right_panel, text="⚙ Settings", 
                                 font=("Arial", 12, "bold"), bg="#2d2d2d", fg="white")
        settings_label.pack(pady=10)
        
        settings_frame = tk.Frame(right_panel, bg="#2d2d2d")
        settings_frame.pack(pady=5, padx=20, fill=tk.X)
        
        # Color selection
        tk.Label(settings_frame, text="Pointer Color:", 
                font=("Arial", 9), bg="#2d2d2d", fg="white").pack(anchor=tk.W, pady=5)
        
        color_frame = tk.Frame(settings_frame, bg="#2d2d2d")
        color_frame.pack(anchor=tk.W, pady=5)
        
        for color in ["green", "blue", "red"]:
            tk.Radiobutton(color_frame, text=color.capitalize(), variable=self.color_choice,
                          value=color, command=self.update_color_settings,
                          font=("Arial", 9), bg="#2d2d2d", fg="white",
                          selectcolor="#1e1e1e", activebackground="#2d2d2d").pack(side=tk.LEFT, padx=5)
        
        # Min area
        tk.Label(settings_frame, text="Min Area Threshold:", 
                font=("Arial", 9), bg="#2d2d2d", fg="white").pack(anchor=tk.W, pady=5)
        
        area_frame = tk.Frame(settings_frame, bg="#2d2d2d")
        area_frame.pack(anchor=tk.W, pady=5, fill=tk.X)
        
        tk.Scale(area_frame, from_=1000, to=10000, orient=tk.HORIZONTAL,
                variable=self.min_area, bg="#2d2d2d", fg="white",
                highlightthickness=0, troughcolor="#1e1e1e").pack(fill=tk.X)
        
        # Show canvas option
        tk.Checkbutton(settings_frame, text="Show Drawing Canvas", 
                      variable=self.show_canvas, command=self.toggle_canvas_display,
                      font=("Arial", 9), bg="#2d2d2d", fg="white",
                      selectcolor="#1e1e1e", activebackground="#2d2d2d").pack(anchor=tk.W, pady=10)
        
        # Separator
        ttk.Separator(right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Prediction display
        prediction_label = tk.Label(right_panel, text="🎯 Prediction", 
                                    font=("Arial", 12, "bold"), bg="#2d2d2d", fg="white")
        prediction_label.pack(pady=10)
        
        self.prediction_frame = tk.Frame(right_panel, bg="#1e1e1e", 
                                        relief=tk.SUNKEN, borderwidth=2)
        self.prediction_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.prediction_text = tk.Label(self.prediction_frame, text="Draw something...", 
                                       font=("Arial", 14, "bold"), bg="#1e1e1e", 
                                       fg="#888", wraplength=280)
        self.prediction_text.pack(pady=20)
        
        self.prediction_image_label = tk.Label(self.prediction_frame, bg="#1e1e1e")
        self.prediction_image_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(right_panel, 
                               text="1. Start camera\n2. Use colored object as pointer\n3. Start drawing\n4. AI will recognize your drawing!",
                               font=("Arial", 8), bg="#2d2d2d", fg="#888",
                               justify=tk.LEFT)
        instructions.pack(side=tk.BOTTOM, pady=10, padx=20)
    
    def start_camera(self):
        """Start the camera"""
        if not self.is_running:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Cannot open camera!")
                return
            
            self.is_running = True
            self.status_indicator.config(fg="green")
            self.status_text.config(text="Running")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.draw_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
    
    def stop_camera(self):
        """Stop the camera"""
        if self.is_running:
            self.is_running = False
            self.is_drawing = False
            if self.camera:
                self.camera.release()
            
            self.status_indicator.config(fg="red")
            self.status_text.config(text="Stopped")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.draw_btn.config(state=tk.DISABLED, text="✏ Start Drawing")
            self.clear_btn.config(state=tk.DISABLED)
            
            # Clear camera display
            self.camera_label.config(image='')
            self.canvas_label.config(image='')
    
    def toggle_drawing(self):
        """Toggle drawing mode"""
        self.is_drawing = not self.is_drawing
        
        if self.is_drawing:
            self.draw_btn.config(text="⏸ Stop Drawing", bg="#FFC107")
            if self.is_shown:
                self.points = deque(maxlen=512)
                self.canvas_img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.is_shown = False
            self.prediction_text.config(text="Drawing...", fg="#2196F3")
            self.prediction_image_label.config(image='')
        else:
            self.draw_btn.config(text="✏ Start Drawing", bg="#2196F3")
            self.process_drawing()
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.points = deque(maxlen=512)
        self.canvas_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.is_shown = False
        self.is_drawing = False
        self.draw_btn.config(text="✏ Start Drawing", bg="#2196F3")
        self.prediction_text.config(text="Draw something...", fg="#888")
        self.prediction_image_label.config(image='')
    
    def toggle_canvas_display(self):
        """Toggle canvas display"""
        if self.show_canvas.get():
            self.canvas_label.pack(padx=10, pady=10)
        else:
            self.canvas_label.pack_forget()
    
    def process_drawing(self):
        """Process the drawn image and make prediction"""
        if len(self.points) == 0:
            return
        
        canvas_gs = cv2.cvtColor(self.canvas_img, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(canvas_gs, 9)
        gaussian = cv2.GaussianBlur(median, (5, 5), 0)
        _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if len(contours):
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            
            if cv2.contourArea(contour) > self.min_area.get():
                x, y, w, h = cv2.boundingRect(contour)
                image = canvas_gs[y:y + h, x:x + w]
                image = cv2.resize(image, (28, 28))
                image = np.array(image, dtype=np.float32)[None, None, :, :]
                image = torch.from_numpy(image)
                
                with torch.no_grad():
                    logits = self.model(image)
                    probabilities = torch.softmax(logits[0], dim=0)
                    self.predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[self.predicted_class].item() * 100
                
                self.is_shown = True
                
                # Update prediction display
                class_name = CLASSES[self.predicted_class]
                self.prediction_text.config(
                    text=f"{class_name.upper()}\nConfidence: {confidence:.1f}%",
                    fg="#4CAF50"
                )
                
                # Display class image if available
                if self.predicted_class < len(self.class_images):
                    img = self.class_images[self.predicted_class]
                    if img is not None:
                        img_resized = cv2.resize(img, (100, 100))
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGBA)
                        img_pil = Image.fromarray(img_rgb)
                        img_tk = ImageTk.PhotoImage(img_pil)
                        self.prediction_image_label.config(image=img_tk)
                        self.prediction_image_label.image = img_tk
            else:
                self.prediction_text.config(text="Drawing too small!\nPlease draw bigger", fg="#f44336")
                self.points = deque(maxlen=512)
                self.canvas_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def update_frame(self):
        """Update camera frame"""
        if self.is_running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            
            if ret:
                frame = cv2.flip(frame, 1)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                kernel = np.ones((5, 5), np.uint8)
                
                mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
                mask = cv2.erode(mask, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours):
                    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
                    ((x, y), radius) = cv2.minEnclosingCircle(contour)
                    cv2.circle(frame, (int(x), int(y)), int(radius), YELLOW_RGB, 2)
                    
                    if self.is_drawing:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                            self.points.appendleft(center)
                            
                            for i in range(1, len(self.points)):
                                if self.points[i - 1] is None or self.points[i] is None:
                                    continue
                                cv2.line(self.canvas_img, self.points[i - 1], 
                                       self.points[i], WHITE_RGB, 5)
                                cv2.line(frame, self.points[i - 1], 
                                       self.points[i], self.color_pointer, 2)
                
                # Display prediction on frame if available
                if self.is_shown and self.predicted_class is not None:
                    cv2.putText(frame, 'Recognized!', (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color_pointer, 3)
                
                # Convert and display camera frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                self.camera_label.config(image=img_tk)
                self.camera_label.image = img_tk
                
                # Display canvas if enabled
                if self.show_canvas.get():
                    canvas_display = 255 - self.canvas_img
                    canvas_rgb = cv2.cvtColor(canvas_display, cv2.COLOR_BGR2RGB)
                    canvas_pil = Image.fromarray(canvas_rgb)
                    canvas_tk = ImageTk.PhotoImage(canvas_pil)
                    self.canvas_label.config(image=canvas_tk)
                    self.canvas_label.image = canvas_tk
        
        # Schedule next update
        self.root.after(10, self.update_frame)
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = QuickDrawGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
