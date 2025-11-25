# camera_handler.py
"""
Module xử lý camera và object detection
"""

import cv2
import numpy as np
from datetime import datetime
import threading


class CameraHandler:
    """Class xử lý camera và phát hiện vật thể"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detected_bbox = None
        self.frame_callbacks = []
        
        # Object detection
        self.object_detector = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
        # Camera settings
        self.frame_width = 1280
        self.frame_height = 720
    
    def start(self):
        """Khởi động camera"""
        if self.is_running:
            return True
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        self.is_running = True
        self._start_capture_thread()
        return True
    
    def stop(self):
        """Dừng camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.detected_bbox = None
    
    def _start_capture_thread(self):
        """Bắt đầu thread capture"""
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()
    
    def _capture_loop(self):
        """Vòng lặp capture frame"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                # Phát hiện vật thể
                self.detected_bbox = self.detect_object(frame)
                
                # Gọi callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame, self.detected_bbox)
                    except Exception as e:
                        print(f"❌ Callback error: {e}")
    
    def detect_object(self, frame):
        """Phát hiện vật thể trong frame"""
        fg_mask = self.object_detector.apply(frame)
        fg_mask[fg_mask == 127] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(
            fg_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 5000:
            return None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            return None
        
        return (x, y, w, h)
    
    def get_current_frame(self):
        """Lấy frame hiện tại"""
        return self.current_frame
    
    def get_cropped_object(self):
        """Lấy vật thể đã crop"""
        if self.current_frame is None:
            return None
        
        if self.detected_bbox:
            x, y, w, h = self.detected_bbox
            return self.current_frame[y:y+h, x:x+w]
        else:
            # Crop vùng giữa màn hình
            h, w = self.current_frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            box_size = 350
            
            x1 = center_x - box_size // 2
            y1 = center_y - box_size // 2
            x2 = center_x + box_size // 2
            y2 = center_y + box_size // 2
            
            return self.current_frame[y1:y2, x1:x2]
    
    def save_current_frame(self, filepath):
        """Lưu frame hiện tại"""
        if self.current_frame is not None:
            cv2.imwrite(filepath, self.current_frame)
            return True
        return False
    
    def add_frame_callback(self, callback):
        """Thêm callback khi có frame mới"""
        self.frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback):
        """Xóa callback"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)


class FrameRenderer:
    """Class render frame với các effects"""
    
    @staticmethod
    def draw_detection_box(frame, bbox):
        """Vẽ khung phát hiện vật thể"""
        if bbox is None:
            return frame
        
        frame = frame.copy()
        x, y, w, h = bbox
        
        # Màu xanh lá
        color = (0, 200, 100)
        thickness = 3
        corner_length = 35
        
        # Vẽ 4 góc bo tròn
        cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
        
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
        
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
        
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)
        
        # Label
        label = "VAT THE PHAT HIEN"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            frame, 
            (x, y - label_h - 15), 
            (x + label_w + 10, y), 
            color, 
            -1
        )
        cv2.putText(
            frame, label, (x + 5, y - 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        # Size info
        size_text = f"{w}x{h}px"
        cv2.putText(
            frame, size_text, (x, y + h + 22), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        
        return frame
    
    @staticmethod
    def draw_center_guide(frame):
        """Vẽ khung hướng dẫn ở giữa"""
        frame = frame.copy()
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        box_size = 350
        
        x1 = center_x - box_size // 2
        y1 = center_y - box_size // 2
        x2 = center_x + box_size // 2
        y2 = center_y + box_size // 2
        
        color = (180, 180, 180)
        thickness = 2
        corner_length = 30
        
        # Vẽ 4 góc
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
        
        # Text hướng dẫn
        cv2.putText(
            frame, "Dat vat pham vao khung", 
            (center_x - 130, y1 - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        
        return frame
    
    @staticmethod
    def draw_status_indicator(frame, auto_scan_enabled):
        """Vẽ trạng thái auto scan"""
        frame = frame.copy()
        
        status_text = "AUTO SCAN: ON" if auto_scan_enabled else "MANUAL MODE"
        status_color = (0, 200, 100) if auto_scan_enabled else (100, 100, 100)
        
        # Background cho status
        (text_w, text_h), _ = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame, (15, 15), (text_w + 35, text_h + 35), 
            status_color, -1
        )
        cv2.putText(
            frame, status_text, (25, 38), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        return frame
    
    @staticmethod
    def render_frame(frame, bbox, auto_scan_enabled):
        """Render frame với tất cả effects"""
        if frame is None:
            return None
        
        # Vẽ detection box hoặc center guide
        if bbox:
            frame = FrameRenderer.draw_detection_box(frame, bbox)
        else:
            frame = FrameRenderer.draw_center_guide(frame)
        
        # Vẽ status indicator
        frame = FrameRenderer.draw_status_indicator(frame, auto_scan_enabled)
        
        return frame


class AutoScanner:
    """Class xử lý auto scan"""
    
    def __init__(self, cooldown=2.0):
        self.enabled = False
        self.cooldown = cooldown
        self.last_scan_time = 0
        self.scan_callback = None
    
    def enable(self):
        """Bật auto scan"""
        self.enabled = True
    
    def disable(self):
        """Tắt auto scan"""
        self.enabled = False
    
    def set_callback(self, callback):
        """Set callback khi cần scan"""
        self.scan_callback = callback
    
    def check_and_scan(self, has_object):
        """Kiểm tra và scan nếu cần"""
        if not self.enabled or not has_object:
            return False
        
        current_time = datetime.now().timestamp()
        if current_time - self.last_scan_time > self.cooldown:
            if self.scan_callback:
                self.scan_callback()
            self.last_scan_time = current_time
            return True
        
        return False