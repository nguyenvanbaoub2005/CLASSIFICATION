# gui_app_refactored.py
"""
GUI đã được refactor - Tách logic camera và UI
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import os
import json
import csv
from datetime import datetime


from classifier import WasteClassifier
from train import train_model, plot_training_history
from data_manager import DataManager
from incremental_train import IncrementalTrainer
from config import PATHS, CLASS_INFO, CLASSES, MODEL_CONFIG
import numpy as np

# Import các module đã tách
from camera_handler import CameraHandler, FrameRenderer, AutoScanner
from ui_components import (
    ModernButton, Card, StatusIndicator, Header, 
    Sidebar, VideoPanel, ResultsPanel, configure_ttk_style
)
from classifier import WasteClassifier
from train import train_model, plot_training_history
from data_manager import DataManager
from incremental_train import IncrementalTrainer
from config import PATHS, CLASS_INFO, CLASSES, MODEL_CONFIG


class WasteClassifierApp:
    """Main Application Class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Hệ Thống Phân Loại Rác Thải AI")
        
        # Window setup
        self._setup_window()
        
        # Colors theme
        self.colors = self._get_colors()
        self.root.configure(bg=self.colors['bg'])
        
        # Initialize components
        self.camera_handler = CameraHandler()
        self.auto_scanner = AutoScanner(cooldown=2.0)
        self.data_manager = DataManager()
        
        # Model
        self.load_model()
        
        # Scan data
        self.scan_history = []
        self.current_result = None
        self.data_save_dir = "scanned_data"
        self._create_save_directories()
        
        # UI Components
        self.ui_components = {}
        
        # Setup UI
        self.setup_ui()
        self.load_scan_history()
        
        # Camera callbacks
        self.camera_handler.add_frame_callback(self.on_camera_frame)
        self.auto_scanner.set_callback(self.on_auto_scan)
    
    def _setup_window(self):
        """Setup window"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.state('zoomed')
    
    def _get_colors(self):
        """Lấy color scheme"""
        return {
            'bg': '#f8f9fa',
            'sidebar': '#ffffff',
            'header': '#ffffff',
            'card': '#ffffff',
            'primary': '#0066cc',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'secondary': '#6c757d',
            'text': '#212529',
            'text_secondary': '#6c757d',
            'border': '#dee2e6',
            'shadow': '#00000010',
        }
    
    def _create_save_directories(self):
        """Tạo thư mục lưu dữ liệu"""
        os.makedirs(self.data_save_dir, exist_ok=True)
        for cls in CLASSES:
            os.makedirs(os.path.join(self.data_save_dir, cls), exist_ok=True)
    
    def load_model(self):
        """Load model"""
        model_path = PATHS['model_save']
        if not os.path.exists(model_path):
            model_path = PATHS['best_model']
        
        try:
            self.classifier = WasteClassifier(model_path)
            self.model_loaded = True
        except:
            self.classifier = None
            self.model_loaded = False
    
    def setup_ui(self):
        """Setup giao diện"""
        # Configure style
        configure_ttk_style()
        
        # Header
        header = Header(
            self.root,
            title="🌿 Phân Loại Rác Thải Thông Minh",
            subtitle="AI-Powered Waste Classification",
            colors=self.colors
        )
        header.pack(fill='x')
        
        # Status indicator
        status = StatusIndicator(header.status_frame, colors=self.colors)
        if self.model_loaded:
            status.set_status("Model Ready", True)
        else:
            status.set_status("Model Not Found", False)
        status.pack()
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left sidebar
        self._setup_sidebar(main)
        
        # Center panel - Camera
        self._setup_camera_panel(main)
        
        # Right panel - Results
        self._setup_results_panel(main)
    
    def _setup_sidebar(self, parent):
        """Setup sidebar"""
        sidebar = Sidebar(parent, colors=self.colors)
        sidebar.pack(side='left', fill='y', padx=(0, 15))
        
        # Menu buttons
        buttons = [
            ("📷 Camera", self.show_camera_mode, self.colors['primary']),
            ("📸 Upload Ảnh", self.upload_image, self.colors['info']),
            ("📹 Xử Lý Video", self.process_video, '#6f42c1'),
            ("📁 Batch", self.batch_classify, self.colors['success']),
            ("🎓 Training", self.show_training_panel, '#fd7e14'),
            ("🔄 Fine-tune", self.incremental_training, '#6f42c1'),
            ("📊 Quản Lý Data", self.show_data_management, self.colors['warning']),
            ("📈 Thống Kê", self.show_statistics, self.colors['info']),
            ("ℹ️ Hướng Dẫn", self.show_guide, self.colors['secondary']),
        ]
        
        for text, command, color in buttons:
            sidebar.add_button(text, command, color)
        
        sidebar.add_spacer()
        
        # Exit button
        sidebar.add_button("🚪 Thoát", self.on_closing, self.colors['secondary'])
        
        self.ui_components['sidebar'] = sidebar
    
    def _setup_camera_panel(self, parent):
        """Setup camera panel"""
        video_panel = VideoPanel(
            parent,
            title="📷 Camera Phát Hiện & Phân Loại",
            colors=self.colors
        )
        video_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Auto scan binding
        video_panel.get_auto_scan_var().trace('w', self._on_auto_scan_toggle)
        
        # Control buttons
        self.btn_camera = video_panel.add_control_button(
            "▶️ Bật Camera",
            self.colors['success'],
            self.toggle_camera
        )
        
        self.btn_scan = video_panel.add_control_button(
            "📸 Scan",
            self.colors['primary'],
            self.manual_scan
        )
        self.btn_scan.config(state='disabled')
        
        self.btn_save_frame = video_panel.add_control_button(
            "💾 Lưu",
            '#6f42c1',
            self.save_current_frame
        )
        self.btn_save_frame.config(state='disabled')
        
        self.ui_components['video_panel'] = video_panel
    
    def _setup_results_panel(self, parent):
        """Setup results panel"""
        results_panel = ResultsPanel(
            parent,
            title="📊 Kết Quả Phân Loại",
            colors=self.colors
        )
        results_panel.pack(side='right', fill='both')
        
        # Action buttons
        self.btn_save_result = results_panel.add_action_button(
            "💾 Lưu Kết Quả",
            '#6f42c1',
            self.save_scan_result
        )
        self.btn_save_result.config(state='disabled')
        
        results_panel.add_action_button(
            "📜 Lịch Sử",
            self.colors['secondary'],
            self.show_history
        )
        
        self.ui_components['results_panel'] = results_panel
        
        # 👉 THÊM DÒNG NÀY ĐỂ KHỞI TẠO self.stats_label
        self.stats_label = results_panel.get_stats_label() 
        
        self.update_statistics()
    
    def _on_auto_scan_toggle(self, *args):
        """Xử lý toggle auto scan"""
        video_panel = self.ui_components['video_panel']
        enabled = video_panel.get_auto_scan_var().get()
        
        if enabled:
            self.auto_scanner.enable()
            print("✅ Bật chế độ tự động quét")
        else:
            self.auto_scanner.disable()
            print("⏸️ Tắt chế độ tự động quét")
    
    # ============ CAMERA METHODS ============
    
    def toggle_camera(self):
        """Bật/tắt camera"""
        if not self.camera_handler.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Khởi động camera"""
        if not self.camera_handler.start():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        
        self.btn_camera.config(text="⏹️ Tắt Camera", bg=self.colors['danger'])
        self.btn_scan.config(state='normal')
        self.btn_save_frame.config(state='normal')
        
        # Start update loop
        self.update_camera_display()
    
    def stop_camera(self):
        """Dừng camera"""
        self.camera_handler.stop()
        self.auto_scanner.disable()
        
        video_panel = self.ui_components['video_panel']
        video_panel.get_auto_scan_var().set(False)
        
        self.btn_camera.config(text="▶️ Bật Camera", bg=self.colors['success'])
        self.btn_scan.config(state='disabled')
        self.btn_save_frame.config(state='disabled')
        
        video_label = video_panel.get_video_label()
        video_label.config(image='')
    
    def on_camera_frame(self, frame, bbox):
        """Callback khi có frame mới từ camera"""
        # Check auto scan
        if bbox:
            self.auto_scanner.check_and_scan(True)
    
    def on_auto_scan(self):
        """Callback khi auto scan trigger"""
        cropped = self.camera_handler.get_cropped_object()
        if cropped is not None and cropped.size > 0:
            temp_path = "temp_auto_scan.jpg"
            cv2.imwrite(temp_path, cropped)
            
            threading.Thread(
                target=self.classify_image_async,
                args=(temp_path, cropped, True),
                daemon=True
            ).start()
    
    def update_camera_display(self):
        """Cập nhật hiển thị camera"""
        if not self.camera_handler.is_running:
            return
        
        frame = self.camera_handler.get_current_frame()
        bbox = self.camera_handler.detected_bbox
        
        if frame is not None:
            # Render frame
            video_panel = self.ui_components['video_panel']
            auto_scan_enabled = video_panel.get_auto_scan_var().get()
            
            rendered_frame = FrameRenderer.render_frame(
                frame.copy(),
                bbox,
                auto_scan_enabled
            )
            
            # Convert và hiển thị
            frame_rgb = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize
            display_height = 480
            aspect_ratio = img.width / img.height
            display_width = int(display_height * aspect_ratio)
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            
            video_label = video_panel.get_video_label()
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_camera_display)
    
    def manual_scan(self):
        """Scan thủ công"""
        cropped = self.camera_handler.get_cropped_object()
        if cropped is None:
            return
        
        temp_path = "temp_manual_scan.jpg"
        cv2.imwrite(temp_path, cropped)
        
        self.classify_image(temp_path, cropped)
    
    def save_current_frame(self):
        """Lưu frame hiện tại"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        
        if self.camera_handler.save_current_frame(filename):
            messagebox.showinfo("Thành công", f"✅ Đã lưu: {filename}")
    
    # ============ CLASSIFICATION METHODS ============
    
    def classify_image(self, image_path, original_image):
        """Phân loại ảnh"""
        if self.classifier is None:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        try:
            result = self.classifier.predict(image_path, return_all=True)
            
            self.current_result = {
                'image_path': image_path,
                'image': original_image,
                'result': result,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'is_auto': False
            }
            
            self.display_result(result)
            self.btn_save_result.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi phân loại: {str(e)}")
    
    def classify_image_async(self, image_path, original_image, is_auto):
        """Phân loại async"""
        try:
            result = self.classifier.predict(image_path, return_all=True)
            
            if result['confidence'] >= 70:
                self.current_result = {
                    'image_path': image_path,
                    'image': original_image,
                    'result': result,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'is_auto': is_auto
                }
                
                self.root.after(0, lambda: self.display_result(result))
                self.root.after(0, lambda: self.btn_save_result.config(state='normal'))
        except Exception as e:
            print(f"❌ Lỗi classify async: {e}")
    
    def display_result(self, result):
        """Hiển thị kết quả"""
        results_panel = self.ui_components['results_panel']
        result_text = results_panel.get_result_text()
        
        predicted_class = result['class']
        confidence = result['confidence']
        info = CLASS_INFO[predicted_class]
        
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        
        # Icon và tên
        result_text.insert(tk.END, f"\n{info['icon']}  ", 'header')
        result_text.insert(tk.END, f"{info['name_vi'].upper()}\n", 'header')
        result_text.insert(tk.END, f"({predicted_class})\n\n", 'info')
        
        # Độ tin cậy
        result_text.insert(tk.END, "🎯 Độ Tin Cậy: ", 'bold')
        
        if result['is_confident']:
            result_text.insert(tk.END, f"{confidence:.1f}% ✅\n", 'success')
        else:
            result_text.insert(tk.END, f"{confidence:.1f}% ⚠️\n", 'warning')
        
        # Progress bar
        bar_length = int(confidence / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        result_text.insert(tk.END, f"{bar}\n\n")
        
        # Hướng dẫn xử lý
        result_text.insert(tk.END, "♻️  Cách Xử Lý:\n", 'bold')
        result_text.insert(tk.END, f"   {info['disposal']}\n\n")
        
        # Ví dụ
        result_text.insert(tk.END, "📝 Ví Dụ:\n", 'bold')
        result_text.insert(tk.END, f"   {', '.join(info['examples'])}\n\n")
        
        # Giá trị tái chế
        result_text.insert(tk.END, "💰 Giá Trị Tái Chế: ", 'bold')
        result_text.insert(tk.END, f"{info['recycling_value']}\n\n")
        
        result_text.insert(tk.END, "─" * 55 + "\n\n")
        
        # Chi tiết xác suất
        result_text.insert(tk.END, "📊 Chi Tiết Các Xác Suất:\n\n", 'bold')
        
        sorted_preds = sorted(
            result['all_predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for cls, prob in sorted_preds:
            icon = CLASS_INFO[cls]['icon']
            bar_length = int(prob / 3)
            bar = "█" * bar_length
            result_text.insert(tk.END, f"{icon} {cls:11s} ")
            result_text.insert(tk.END, f"{bar:33s} {prob:5.1f}%\n")
        
        result_text.config(state='disabled')
    
    # ============ SAVE & HISTORY METHODS ============
    
    def save_scan_result(self):
        """Lưu kết quả scan"""
        if not hasattr(self, 'current_result'):
            return
        
        result = self.current_result['result']
        predicted_class = result['class']
        confidence = result['confidence']
        
        if confidence < 80:
            response = messagebox.askyesno(
                "Xác nhận",
                f"Độ tin cậy thấp ({confidence:.1f}%).\nBạn có chắc muốn lưu?"
            )
            if not response:
                return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{predicted_class}_{timestamp}_{confidence:.0f}.jpg"
        save_path = os.path.join(self.data_save_dir, predicted_class, filename)
        
        cv2.imwrite(save_path, self.current_result['image'])
        
        metadata = {
            'class': predicted_class,
            'confidence': confidence,
            'timestamp': self.current_result['timestamp'],
            'all_predictions': result['all_predictions'],
            'is_auto_scan': self.current_result.get('is_auto', False)
        }
        
        json_path = save_path.replace('.jpg', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.scan_history.append(metadata)
        self.save_scan_history()
        self.update_statistics()
        
        messagebox.showinfo("Thành công", f"✅ Đã lưu kết quả!\n\n{save_path}")
        self.btn_save_result.config(state='disabled')
    
    # def save_scan_history(self):
    #     """Lưu lịch sử"""
    #     history_path = os.path.join(self.data_save_dir, 'scan_history.json')
    #     with open(history_path, 'w', encoding='utf-8') as f:
    #         json.dump(self.scan_history, f, indent=2, ensure_ascii=False)
    
    # def load_scan_history(self):
    #     """Load lịch sử"""
    #     history_path = os.path.join(self.data_save_dir, 'scan_history.json')
    #     if os.path.exists(history_path):
    #         try:
    #             with open(history_path, 'r', encoding='utf-8') as f:
    #                 self.scan_history = json.load(f)
    #         except:
    #             self.scan_history = []
    #     else:
    #         self.scan_history = []
    
    # def update_statistics(self):
    #     """Cập nhật thống kê"""
    #     results_panel = self.ui_components['results_panel']
    #     stats_label = results_panel.get_stats_label()
        
    #     stats = {cls: 0 for cls in CLASSES}
    #     high_conf_count = 0
    #     auto_count = 0
        
    #     for item in self.scan_history:
    #         stats[item['class']] += 1
    #         if item['confidence'] >= 80:
    #             high_conf_count += 1
    #         if item.get('is_auto_scan', False):
    #             auto_count += 1
        
    #     total = len(self.scan_history)
        
    #     if total == 0:
    #         stats_label.config(text="Chưa có dữ liệu")
    #         return
        
    #     text = f"📊 Tổng: {total} lần scan\n"
    #     text += f"✅ Tin cậy cao: {high_conf_count}/{total}\n"
    #     text += f"🤖 Auto scan: {auto_count}/{total}\n\n"
        
    #     # Top 3 classes
    #     sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
    #     for cls, count in sorted_stats:
    #         if count > 0:
    #             icon = CLASS_INFO[cls]['icon']
    #             pct = (count / total * 100)
    #             text += f"{icon} {cls}: {count} ({pct:.0f}%)\n"
        
    #     stats_label.config(text=text)
    
    # ============ MENU ACTIONS ============
    
    def show_camera_mode(self):
        """Chuyển về camera mode"""
        messagebox.showinfo(
            "Camera Mode",
            "📷 Chế độ camera đang hiển thị ở màn hình chính!\n\n" +
            "• Nhấn '▶️ Bật Camera' để bắt đầu\n" +
            "• Bật 'Tự động quét' để scan liên tục\n" +
            "• Khung xanh tự động theo dõi vật thể"
        )
    
    def upload_image(self):
        """Upload ảnh"""
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            img = cv2.imread(file_path)
            self.classify_image(file_path, img)
    
    def process_video(self):
        """Xử lý video"""
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        video_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if not video_path:
            return
        
        save_output = messagebox.askyesno("Lưu video?", "Bạn có muốn lưu video kết quả không?")
        
        output_path = None
        if save_output:
            output_path = filedialog.asksaveasfilename(
                title="Lưu video",
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")]
            )
        
        threading.Thread(
            target=self.process_video_thread,
            args=(video_path, output_path),
            daemon=True
        ).start()
    
    def process_video_thread(self, video_path, output_path):
        """Xử lý video thread"""
        try:
            cam_classifier = CameraClassifier(PATHS['model_save'])
            cam_classifier.classify_video_file(video_path, output_path)
            
            self.root.after(0, lambda: messagebox.showinfo("Thành công", "✅ Đã xử lý video!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi xử lý video: {str(e)}"))
    
    def batch_classify(self):
        """Phân loại batch"""
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        
        if not folder_path:
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy ảnh nào!")
            return
        
        threading.Thread(target=self.batch_classify_thread, args=(image_files,), daemon=True).start()
    
    def batch_classify_thread(self, image_files):
        """Batch classify thread"""
        try:
            results = self.classifier.predict_batch(image_files)
            self.root.after(0, lambda: self.show_batch_results(results))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi batch: {str(e)}"))
    
    def show_batch_results(self, results):
        """Hiển thị kết quả batch"""
        window = tk.Toplevel(self.root)
        window.title("📁 Kết Quả Batch")
        window.geometry("1100x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=70)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📊 Kết Quả Phân Loại Batch",
            font=('Segoe UI', 20, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['primary']
        ).pack(pady=20)
        
        # Treeview
        tree_frame = tk.Frame(window, bg=self.colors['bg'])
        tree_frame.pack(fill='both', expand=True, padx=30, pady=20)
        
        columns = ('STT', 'File', 'Loại', 'Confidence', 'Status')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=22)
        
        for col in columns:
            tree.heading(col, text=col)
        
        tree.column('STT', width=60)
        tree.column('File', width=400)
        tree.column('Loại', width=250)
        tree.column('Confidence', width=130)
        tree.column('Status', width=100)
        
        for i, item in enumerate(results, 1):
            result = item['result']
            filename = os.path.basename(item['image'])
            icon = CLASS_INFO[result['class']]['icon']
            status = "✅ Cao" if result['is_confident'] else "⚠️ Thấp"
            
            tree.insert('', 'end', values=(
                i, filename,
                f"{icon} {result['class_name_vi']}",
                f"{result['confidence']:.1f}%",
                status
            ))
        
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Buttons
        btn_frame = tk.Frame(window, bg=self.colors['bg'])
        btn_frame.pack(pady=20)
        
        ModernButton(
            btn_frame, text="💾 Lưu CSV", bg=self.colors['primary'], fg='white',
            command=lambda: self.save_batch_csv(results)
        ).pack(side='left', padx=10)
        
        ModernButton(
            btn_frame, text="🚪 Đóng", bg=self.colors['secondary'], fg='white',
            command=window.destroy
        ).pack(side='left', padx=10)
    
    def save_batch_csv(self, results):
        """Lưu CSV"""
        file_path = filedialog.asksaveasfilename(
            title="Lưu CSV", defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['STT', 'File', 'Class', 'Class_VI', 'Confidence', 'Status'])
                
                for i, item in enumerate(results, 1):
                    result = item['result']
                    status = "High" if result['is_confident'] else "Low"
                    writer.writerow([
                        i, os.path.basename(item['image']),
                        result['class'], result['class_name_vi'],
                        f"{result['confidence']:.2f}", status
                    ])
            
            messagebox.showinfo("Thành công", f"✅ Đã lưu: {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi lưu CSV: {str(e)}")
    
    def show_training_panel(self):
        """Panel training"""
        window = tk.Toplevel(self.root)
        window.title("🎓 Training Model")
        window.geometry("900x700")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header, text="🎓 Training Model Mới",
            font=('Segoe UI', 22, 'bold'),
            bg=self.colors['card'], fg=self.colors['primary']
        ).pack(pady=25)
        
        # Form
        form = self.create_card(window)
        form.pack(fill='both', expand=True, padx=30, pady=20)
        
        # Train dir
        tk.Label(form, text="📁 Thư mục Training:", font=('Segoe UI', 12),
                bg=self.colors['card'], fg=self.colors['text']).pack(pady=(20, 5), anchor='w', padx=30)
        
        train_frame = tk.Frame(form, bg=self.colors['card'])
        train_frame.pack(fill='x', padx=30, pady=5)
        
        train_entry = tk.Entry(train_frame, font=('Segoe UI', 11), width=60,
                              relief='solid', bd=1)
        train_entry.pack(side='left', ipady=8, padx=(0, 10))
        
        ModernButton(train_frame, text="Browse", bg=self.colors['info'], fg='white',
                    command=lambda: train_entry.insert(0, filedialog.askdirectory())).pack()
        
        # Val dir
        tk.Label(form, text="📁 Thư mục Validation:", font=('Segoe UI', 12),
                bg=self.colors['card'], fg=self.colors['text']).pack(pady=(15, 5), anchor='w', padx=30)
        
        val_frame = tk.Frame(form, bg=self.colors['card'])
        val_frame.pack(fill='x', padx=30, pady=5)
        
        val_entry = tk.Entry(val_frame, font=('Segoe UI', 11), width=60,
                            relief='solid', bd=1)
        val_entry.pack(side='left', ipady=8, padx=(0, 10))
        
        ModernButton(val_frame, text="Browse", bg=self.colors['info'], fg='white',
                    command=lambda: val_entry.insert(0, filedialog.askdirectory())).pack()
        
        # Epochs
        tk.Label(form, text="⏱️ Số Epochs:", font=('Segoe UI', 12),
                bg=self.colors['card'], fg=self.colors['text']).pack(pady=(15, 5), anchor='w', padx=30)
        
        epochs_entry = tk.Entry(form, font=('Segoe UI', 11), width=20, relief='solid', bd=1)
        epochs_entry.insert(0, "50")
        epochs_entry.pack(anchor='w', padx=30, pady=5, ipady=8)
        
        # Transfer learning
        transfer_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            form, text="🔄 Sử dụng Transfer Learning",
            variable=transfer_var, font=('Segoe UI', 12),
            bg=self.colors['card'], fg=self.colors['text'],
            selectcolor=self.colors['card'], activebackground=self.colors['card']
        ).pack(pady=20, anchor='w', padx=30)
        
        # Button
        ModernButton(
            form, text="🚀 Bắt Đầu Training",
            bg=self.colors['success'], fg='white', width=25,
            command=lambda: self.start_training(
                train_entry.get(), val_entry.get(),
                int(epochs_entry.get()), transfer_var.get(), window
            )
        ).pack(pady=30)
    
    def start_training(self, train_dir, val_dir, epochs, use_transfer, window):
        """Bắt đầu training"""
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            messagebox.showerror("Lỗi", "Thư mục không tồn tại!")
            return
        
        window.destroy()
        threading.Thread(
            target=self.training_thread,
            args=(train_dir, val_dir, epochs, use_transfer),
            daemon=True
        ).start()
        
        messagebox.showinfo("Training", "Training đã bắt đầu!\nKiểm tra console.")
    
    def training_thread(self, train_dir, val_dir, epochs, use_transfer):
        """Training thread"""
        try:
            model, history = train_model(train_dir, val_dir, epochs=epochs,
                                        use_transfer_learning=use_transfer)
            plot_training_history(history)
            self.load_model()
            
            self.root.after(0, lambda: messagebox.showinfo(
                "Thành công", "✅ Training hoàn tất!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi: {str(e)}"))
    
    def incremental_training(self):
        """Incremental training"""
        trainer = IncrementalTrainer()
        ready, stats = trainer.check_data_ready()
        
        if not ready:
            msg = "❌ Dữ liệu chưa đủ!\n\nCần ít nhất 20 mẫu chất lượng cao/class.\n\n"
            for cls, data in stats['by_class'].items():
                msg += f"{cls}: {data['high_confidence']} mẫu\n"
            
            messagebox.showwarning("Cảnh báo", msg)
            return
        
        if messagebox.askyesno("Xác nhận", f"✅ Dữ liệu sẵn sàng!\n\nTổng: {stats['total']}\nBắt đầu training?"):
            threading.Thread(target=self.incremental_training_thread,
                           args=(trainer,), daemon=True).start()
            messagebox.showinfo("Training", "Incremental training đã bắt đầu!")
    
    def incremental_training_thread(self, trainer):
        """Incremental training thread"""
        try:
            trainer.prepare_incremental_data()
            model, history = trainer.train_incremental(epochs=20, fine_tune=True)
            self.load_model()
            
            self.root.after(0, lambda: messagebox.showinfo("Thành công", "✅ Hoàn tất!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi: {str(e)}"))
    
    def show_data_management(self):
        """Quản lý dữ liệu"""
        window = tk.Toplevel(self.root)
        window.title("📊 Quản Lý Dữ Liệu")
        window.geometry("1000x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="📊 Quản Lý Dữ Liệu Training",
                font=('Segoe UI', 22, 'bold'),
                bg=self.colors['card'], fg=self.colors['primary']).pack(pady=25)
        
        # Stats
        stats = self.data_manager.get_scanned_stats()
        
        overview = f"""
📈 TỔNG QUAN
{'─'*70}
Tổng số mẫu: {stats['total']}
Chất lượng cao (≥80%): {stats['high_confidence']}
Tỷ lệ: {stats['high_confidence']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%

📋 CHI TIẾT THEO CLASS
{'─'*70}
"""
        
        for cls in CLASSES:
            data = stats['by_class'][cls]
            icon = CLASS_INFO[cls]['icon']
            overview += f"{icon} {cls:12s}: {data['count']:4d} (Cao: {data['high_confidence']}, Thấp: {data['low_confidence']})\n"
        
        text = scrolledtext.ScrolledText(window, font=('Consolas', 11),
                                        bg='#f8f9fa', fg=self.colors['text'],
                                        wrap='word', height=20, relief='solid', bd=1)
        text.pack(fill='both', expand=True, padx=30, pady=20)
        text.insert(1.0, overview)
        text.config(state='disabled')
        
        # Buttons
        btn_frame = tk.Frame(window, bg=self.colors['bg'])
        btn_frame.pack(pady=20)
        
        ModernButton(btn_frame, text="📦 Chuẩn Bị", bg=self.colors['primary'], fg='white',
                    command=self.prepare_dataset).pack(side='left', padx=8)
        ModernButton(btn_frame, text="📤 Export", bg=self.colors['success'], fg='white',
                    command=self.export_high_quality).pack(side='left', padx=8)
        ModernButton(btn_frame, text="🗑️ Xóa", bg=self.colors['danger'], fg='white',
                    command=self.clean_low_quality).pack(side='left', padx=8)
    
    def prepare_dataset(self):
        """Chuẩn bị dataset"""
        if messagebox.askyesno("Xác nhận", "Chuẩn bị dữ liệu cho training?"):
            try:
                self.data_manager.prepare_training_data(min_confidence=80)
                messagebox.showinfo("Thành công", "✅ Đã chuẩn bị dataset!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")
    
    def export_high_quality(self):
        """Export chất lượng cao"""
        output_dir = filedialog.askdirectory(title="Chọn thư mục lưu")
        if output_dir:
            try:
                self.data_manager.export_high_quality_data(output_dir, min_confidence=90)
                messagebox.showinfo("Thành công", f"✅ Đã export!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")
    
    def clean_low_quality(self):
        """Xóa chất lượng thấp"""
        if messagebox.askyesno("Cảnh báo", "⚠️ Xóa ảnh ≤60%? Không thể hoàn tác!"):
            try:
                self.data_manager.clean_low_quality_data(max_confidence=60)
                messagebox.showinfo("Thành công", "✅ Đã xóa!")
                self.update_statistics()
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")
    
    def show_statistics(self):
        """Thống kê"""
        stats = self.data_manager.get_scanned_stats()
        
        msg = f"""📊 THỐNG KÊ CHI TIẾT

{'═'*50}
TỔNG QUAN
{'═'*50}
• Tổng mẫu: {stats['total']}
• Chất lượng cao: {stats['high_confidence']}
• Tỷ lệ: {stats['high_confidence']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%

{'═'*50}
CHI TIẾT
{'═'*50}
"""
        
        for cls in CLASSES:
            data = stats['by_class'][cls]
            icon = CLASS_INFO[cls]['icon']
            msg += f"\n{icon} {CLASS_INFO[cls]['name_vi']}:\n"
            msg += f"   Tổng: {data['count']}, Cao: {data['high_confidence']}, Thấp: {data['low_confidence']}\n"
        
        messagebox.showinfo("Thống Kê", msg)
    
    def show_guide(self):
        """Hướng dẫn"""
        window = tk.Toplevel(self.root)
        window.title("ℹ️ Hướng Dẫn")
        window.geometry("1000x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="📖 Hướng Dẫn Sử Dụng",
                font=('Segoe UI', 22, 'bold'),
                bg=self.colors['card'], fg=self.colors['primary']).pack(pady=25)
        
        guide = """
╔═══════════════════════════════════════════════════════════════╗
║                    🎯 HƯỚNG DẪN SỬ DỤNG                        ║
╚═══════════════════════════════════════════════════════════════╝

📷 CAMERA SCAN
──────────────────────────────────────────────────────────────
1. Nhấn "▶️ Bật Camera"
2. Đặt vật phẩm vào khung
3. Hệ thống tự động phát hiện và DI CHUYỂN KHUNG XANH
4. Bật "🤖 Tự động quét" để scan liên tục (mỗi 2 giây)
5. Hoặc nhấn "📸 Scan" để scan thủ công
6. Xem kết quả bên phải và lưu nếu cần

📸 UPLOAD & BATCH
──────────────────────────────────────────────────────────────
• Upload: Chọn 1 ảnh để phân loại
• Batch: Chọn thư mục nhiều ảnh, xem kết quả bảng, lưu CSV

🎓 TRAINING
──────────────────────────────────────────────────────────────
• Training: Train model mới từ dataset có sẵn
• Fine-tune: Cập nhật model với dữ liệu đã scan (≥20 mẫu/class)

📊 QUẢN LÝ DỮ LIỆU
──────────────────────────────────────────────────────────────
• Xem thống kê dữ liệu đã scan
• Chuẩn bị dataset (auto chia 80/20)
• Export dữ liệu chất lượng cao (≥90%)
• Xóa dữ liệu kém (≤60%)

💡 TIPS
──────────────────────────────────────────────────────────────
✓ Khung xanh tự động theo dõi vật thể
✓ Chỉ lưu ảnh confidence ≥80%
✓ Dùng Fine-tune để cải thiện model liên tục
✓ Auto scan cooldown 2 giây tránh spam

⚙️ YÊU CẦU HỆ THỐNG
──────────────────────────────────────────────────────────────
• Python 3.7+
• TensorFlow 2.x
• OpenCV
• Camera (cho real-time)

──────────────────────────────────────────────────────────────
Happy Classifying! 🌿
──────────────────────────────────────────────────────────────
"""
        
        text = scrolledtext.ScrolledText(
            window,
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg=self.colors['text'],
            wrap='word',
            relief='solid',
            bd=1
        )
        text.pack(fill='both', expand=True, padx=30, pady=(0, 20))
        text.insert(1.0, guide)
        text.config(state='disabled')
        
        ModernButton(
            window,
            text="🚪 Đóng",
            bg=self.colors['secondary'],
            fg='white',
            command=window.destroy
        ).pack(pady=20)
    
    def show_camera_mode(self):
        """Chuyển về camera mode"""
        messagebox.showinfo(
            "Camera Mode",
            "📷 Chế độ camera đang hiển thị ở màn hình chính!\n\n" +
            "• Nhấn '▶️ Bật Camera' để bắt đầu\n" +
            "• Bật 'Tự động quét' để scan liên tục\n" +
            "• Khung xanh tự động theo dõi vật thể"
        )
    
    def show_history(self):
        """Hiển thị lịch sử"""
        window = tk.Toplevel(self.root)
        window.title("📜 Lịch Sử Scan")
        window.geometry("1100x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📜 Lịch Sử Phân Loại",
            font=('Segoe UI', 22, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['primary']
        ).pack(pady=25)
        
        # Treeview
        tree_frame = tk.Frame(window, bg=self.colors['bg'])
        tree_frame.pack(fill='both', expand=True, padx=30, pady=20)
        
        columns = ('STT', 'Loại', 'Confidence', 'Thời gian', 'Mode')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=25)
        
        tree.heading('STT', text='STT')
        tree.heading('Loại', text='Loại Rác')
        tree.heading('Confidence', text='Độ Tin Cậy')
        tree.heading('Thời gian', text='Thời Gian')
        tree.heading('Mode', text='Chế Độ')
        
        tree.column('STT', width=60)
        tree.column('Loại', width=280)
        tree.column('Confidence', width=130)
        tree.column('Thời gian', width=180)
        tree.column('Mode', width=120)
        
        # Thêm dữ liệu
        for i, item in enumerate(reversed(self.scan_history), 1):
            icon = CLASS_INFO[item['class']]['icon']
            mode = "🤖 Auto" if item.get('is_auto_scan', False) else "👤 Manual"
            
            tree.insert('', 'end', values=(
                i,
                f"{icon} {CLASS_INFO[item['class']]['name_vi']}",
                f"{item['confidence']:.1f}%",
                item['timestamp'],
                mode
            ))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Close button
        ModernButton(
            window,
            text="🚪 Đóng",
            bg=self.colors['secondary'],
            fg='white',
            command=window.destroy
        ).pack(pady=20)
    
    def save_scan_history(self):
        """Lưu lịch sử"""
        history_path = os.path.join(self.data_save_dir, 'scan_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.scan_history, f, indent=2, ensure_ascii=False)
    
    def load_scan_history(self):
        """Load lịch sử"""
        history_path = os.path.join(self.data_save_dir, 'scan_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.scan_history = json.load(f)
            except:
                self.scan_history = []
        else:
            self.scan_history = []
    
    def update_statistics(self):
        """Cập nhật thống kê"""
        stats = {cls: 0 for cls in CLASSES}
        high_conf_count = 0
        auto_count = 0
        
        for item in self.scan_history:
            stats[item['class']] += 1
            if item['confidence'] >= 80:
                high_conf_count += 1
            if item.get('is_auto_scan', False):
                auto_count += 1
        
        total = len(self.scan_history)
        
        if total == 0:
            self.stats_label.config(text="Chưa có dữ liệu")
            return
        
        text = f"📊 Tổng: {total} lần scan\n"
        text += f"✅ Tin cậy cao: {high_conf_count}/{total}\n"
        text += f"🤖 Auto scan: {auto_count}/{total}\n\n"
        
        # Top 3 classes
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
        for cls, count in sorted_stats:
            if count > 0:
                icon = CLASS_INFO[cls]['icon']
                pct = (count / total * 100)
                text += f"{icon} {cls}: {count} ({pct:.0f}%)\n"
        
        self.stats_label.config(text=text)
    
    def on_closing(self):
        """Xử lý đóng cửa sổ"""
        if self.camera_running:
            self.stop_camera()
        
        if messagebox.askokcancel("Thoát", "Bạn có chắc muốn thoát?"):
            self.root.destroy()



def main():
    """Main function"""
    root = tk.Tk()
    app = WasteClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()