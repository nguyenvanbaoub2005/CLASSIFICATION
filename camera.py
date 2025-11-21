# camera.py
"""
Module xử lý camera và phân loại real-time
"""

import cv2
import os
from classifier import WasteClassifier
from config import PATHS, CLASS_INFO, COLORS


class CameraClassifier:
    """Class xử lý phân loại từ camera"""
    
    def __init__(self, model_path=None):
        """
        Khởi tạo camera classifier
        
        Args:
            model_path: Đường dẫn model
        """
        if model_path is None:
            model_path = PATHS['model_save']
            if not os.path.exists(model_path):
                model_path = PATHS['best_model']
        
        self.classifier = WasteClassifier(model_path)
        self.cap = None
        self.is_running = False
        
    def start_camera(self, camera_id=0):
        """
        Khởi động camera
        
        Args:
            camera_id: ID của camera (0 cho camera mặc định)
        
        Returns:
            bool: Thành công hay không
        """
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print("❌ Không thể mở camera!")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_running = True
        print("✅ Camera đã sẵn sàng!")
        return True
    
    def stop_camera(self):
        """Dừng camera"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        print("🛑 Đã đóng camera")
    
    def put_text_with_background(self, img, text, position, font_scale=0.7, 
                                 thickness=2, bg_color=(0, 0, 0), 
                                 text_color=(255, 255, 255)):
        """
        Vẽ text với background      
        
        Args:
            img: Image frame
            text: Text cần vẽ
            position: Vị trí (x, y)
            font_scale: Kích thước font
            thickness: Độ dày
            bg_color: Màu background
            text_color: Màu text
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Lấy kích thước text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Vẽ background
        x, y = position
        padding = 10
        cv2.rectangle(
            img,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1
        )
        
        # Vẽ text
        cv2.putText(
            img, text, position, font, font_scale, 
            text_color, thickness, cv2.LINE_AA
        )
    
    def draw_classification_result(self, frame, result):
        """
        Vẽ kết quả phân loại lên frame
        
        Args:
            frame: Video frame
            result: Kết quả phân loại
        
        Returns:
            frame: Frame đã vẽ kết quả
        """
        if result is None:
            return frame
        
        predicted_class = result['class']
        confidence = result['confidence']
        info = CLASS_INFO[predicted_class]
        
        # Màu sắc dựa trên confidence
        if confidence >= 80:
            color = (0, 255, 0)  # Green
        elif confidence >= 60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Vẽ header
        header = f"{info['icon']} {info['name_vi'].upper()}"
        self.put_text_with_background(
            frame, header, (20, 50),
            font_scale=1.0, thickness=3,
            bg_color=(0, 0, 0), text_color=color
        )
        
        # Vẽ confidence
        conf_text = f"Tin cay: {confidence:.1f}%"
        self.put_text_with_background(
            frame, conf_text, (20, 100),
            font_scale=0.8, thickness=2,
            bg_color=(0, 0, 0), text_color=color
        )
        
        # Vẽ hướng dẫn xử lý
        disposal = info['disposal']
        self.put_text_with_background(
            frame, disposal, (20, 150),
            font_scale=0.6, thickness=2,
            bg_color=(0, 0, 0), text_color=(255, 255, 255)
        )
        
        # Vẽ các xác suất khác (top 3)
        if 'all_predictions' in result:
            sorted_preds = sorted(
                result['all_predictions'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            y_pos = frame.shape[0] - 150
            for i, (cls, prob) in enumerate(sorted_preds):
                cls_info = CLASS_INFO[cls]
                text = f"{cls_info['icon']} {cls}: {prob:.1f}%"
                self.put_text_with_background(
                    frame, text, (20, y_pos + i * 40),
                    font_scale=0.6, thickness=2,
                    bg_color=(50, 50, 50), text_color=(200, 200, 200)
                )
        
        return frame
    
    def run_interactive(self):
        """Chạy chế độ tương tác với camera"""
        if not self.is_running:
            print("❌ Camera chưa được khởi động!")
            return
        
        print("\n" + "="*70)
        print("📷 CAMERA PHÂN LOẠI RÁC THẢI")
        print("="*70)
        print("Điều khiển:")
        print("  SPACE  - Chụp và phân loại")
        print("  C      - Phân loại liên tục (toggle)")
        print("  S      - Lưu ảnh")
        print("  Q/ESC  - Thoát")
        print("="*70 + "\n")
        
        continuous_mode = False
        last_result = None
        save_counter = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Không thể đọc frame từ camera!")
                break
            
            # Flip frame cho tự nhiên hơn
            frame = cv2.flip(frame, 1)
            
            # Vẽ hướng dẫn
            if not continuous_mode:
                cv2.putText(
                    frame, "Nhan SPACE de phan loai", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
            else:
                cv2.putText(
                    frame, "CHE DO LIEN TUC", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            
            # Phân loại liên tục
            if continuous_mode:
                try:
                    temp_path = PATHS['temp_image']
                    cv2.imwrite(temp_path, frame)
                    last_result = self.classifier.predict(temp_path, return_all=True)
                    frame = self.draw_classification_result(frame, last_result)
                except Exception as e:
                    print(f"⚠️  Lỗi phân loại: {str(e)}")
            elif last_result:
                # Vẽ kết quả cuối cùng
                frame = self.draw_classification_result(frame, last_result)
            
            # Hiển thị frame
            cv2.imshow('Waste Classifier Camera', frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - Chụp và phân loại
                print("\n📸 Đang chụp và phân loại...")
                temp_path = PATHS['temp_image']
                cv2.imwrite(temp_path, frame)
                
                try:
                    last_result = self.classifier.predict(temp_path, return_all=True)
                    self.classifier.display_result(temp_path, last_result)
                except Exception as e:
                    print(f"❌ Lỗi: {str(e)}")
                    last_result = None
            
            elif key == ord('c') or key == ord('C'):  # Toggle continuous mode
                continuous_mode = not continuous_mode
                status = "BẬT" if continuous_mode else "TẮT"
                print(f"🔄 Chế độ liên tục: {status}")
            
            elif key == ord('s') or key == ord('S'):  # Lưu ảnh
                save_counter += 1
                filename = f"captured_{save_counter}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Đã lưu ảnh: {filename}")
            
            elif key == ord('q') or key == ord('Q') or key == 27:  # Q hoặc ESC
                break
        
        self.stop_camera()
    
    def classify_video_file(self, video_path, output_path=None, interval=30):
        """
        Phân loại từ file video
        
        Args:
            video_path: Đường dẫn video
            output_path: Đường dẫn lưu video kết quả (optional)
            interval: Phân loại mỗi N frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            return
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Đang xử lý video:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Setup video writer nếu cần
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        last_result = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Phân loại mỗi interval frames
            if frame_count % interval == 0:
                temp_path = PATHS['temp_image']
                cv2.imwrite(temp_path, frame)
                try:
                    last_result = self.classifier.predict(temp_path, return_all=False)
                    print(f"Frame {frame_count}/{total_frames}: "
                          f"{last_result['class_name_vi']} "
                          f"({last_result['confidence']:.1f}%)")
                except:
                    pass
            
            # Vẽ kết quả
            if last_result:
                frame = self.draw_classification_result(frame, last_result)
            
            # Ghi video
            if writer:
                writer.write(frame)
            
            # Hiển thị
            cv2.imshow('Processing Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
            print(f"✅ Video đã lưu tại: {output_path}")
        
        cv2.destroyAllWindows()


def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          📷 CAMERA PHÂN LOẠI RÁC THẢI                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    cam_classifier = CameraClassifier()
    
    print("\nChọn chế độ:")
    print("1. Camera real-time")
    print("2. Xử lý video file")
    
    choice = input("\nNhập lựa chọn (1/2): ").strip()
    
    if choice == '1':
        if cam_classifier.start_camera():
            cam_classifier.run_interactive()
    
    elif choice == '2':
        video_path = input("Đường dẫn video: ").strip()
        if os.path.exists(video_path):
            output_path = input("Đường dẫn lưu kết quả (Enter để bỏ qua): ").strip()
            output_path = output_path if output_path else None
            cam_classifier.classify_video_file(video_path, output_path)
        else:
            print("❌ File không tồn tại!")


if __name__ == "__main__":
    main()