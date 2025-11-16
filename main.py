# main.py
"""
File chính để chạy hệ thống phân loại rác thải - Menu tổng hợp tất cả chức năng
"""

import os
import sys
from classifier import WasteClassifier
from camera import CameraClassifier
from train import train_model, plot_training_history
from config import PATHS, CLASSES
import matplotlib.pyplot as plt
from PIL import Image


def print_banner():
    """In banner chào mừng"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       🤖 HỆ THỐNG PHÂN LOẠI RÁC THẢI BẰNG AI - PYTHON       ║
    ║                                                               ║
    ║              Sử dụng Deep Learning (CNN)                      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """In menu chính"""
    print("\n" + "="*70)
    print("📋 MENU CHÍNH")
    print("="*70)
    print("1. 📸 Phân loại từ ảnh (File)")
    print("2. 📷 Phân loại từ Camera (Real-time)")
    print("3. 📹 Phân loại từ Video")
    print("4. 📁 Phân loại nhiều ảnh (Batch)")
    print("5. 🎓 Huấn luyện Model mới")
    print("6. 📊 Xem thông tin Model")
    print("7. ℹ️  Hướng dẫn sử dụng")
    print("0. 🚪 Thoát")
    print("="*70)


def classify_from_image():
    """Phân loại từ file ảnh"""
    print("\n📸 PHÂN LOẠI TỪ ẢNH")
    print("-" * 70)
    
    # Load model
    model_path = PATHS['model_save']
    if not os.path.exists(model_path):
        model_path = PATHS['best_model']
    
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model đã huấn luyện!")
        print("   Vui lòng huấn luyện model trước (chọn option 5)")
        return
    
    classifier = WasteClassifier(model_path)
    
    # Nhập đường dẫn ảnh
    image_path = input("\n📁 Nhập đường dẫn ảnh: ").strip()
    
    if not os.path.exists(image_path):
        print("❌ File không tồn tại!")
        return
    
    try:
        # Phân loại
        print("\n🔍 Đang phân tích...")
        result = classifier.predict(image_path)
        
        # Hiển thị kết quả
        classifier.display_result(image_path, result)
        
        # Hỏi có muốn hiển thị ảnh không
        show = input("Hiển thị ảnh? (y/n): ").strip().lower()
        if show == 'y':
            img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(
                f"{result['class_name_vi']}\nConfidence: {result['confidence']:.2f}%",
                fontsize=16,
                fontweight='bold'
            )
            plt.tight_layout()
            plt.show()
    
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")


def classify_from_camera():
    """Phân loại từ camera"""
    print("\n📷 PHÂN LOẠI TỪ CAMERA")
    print("-" * 70)
    
    # Load model
    model_path = PATHS['model_save']
    if not os.path.exists(model_path):
        model_path = PATHS['best_model']
    
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model đã huấn luyện!")
        return
    
    cam_classifier = CameraClassifier(model_path)
    
    if cam_classifier.start_camera():
        cam_classifier.run_interactive()


def classify_from_video():
    """Phân loại từ video file"""
    print("\n📹 PHÂN LOẠI TỪ VIDEO")
    print("-" * 70)
    
    # Load model
    model_path = PATHS['model_save']
    if not os.path.exists(model_path):
        model_path = PATHS['best_model']
    
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model đã huấn luyện!")
        return
    
    video_path = input("\n📁 Đường dẫn video: ").strip()
    
    if not os.path.exists(video_path):
        print("❌ File không tồn tại!")
        return
    
    save_output = input("Lưu video kết quả? (y/n): ").strip().lower()
    output_path = None
    
    if save_output == 'y':
        output_path = input("Đường dẫn lưu (ví dụ: output.mp4): ").strip()
        if not output_path:
            output_path = "classified_video.mp4"
    
    cam_classifier = CameraClassifier(model_path)
    cam_classifier.classify_video_file(video_path, output_path)


def classify_batch():
    """Phân loại nhiều ảnh"""
    print("\n📁 PHÂN LOẠI BATCH")
    print("-" * 70)
    
    # Load model
    model_path = PATHS['model_save']
    if not os.path.exists(model_path):
        model_path = PATHS['best_model']
    
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model đã huấn luyện!")
        return
    
    classifier = WasteClassifier(model_path)
    
    folder_path = input("\n📁 Đường dẫn thư mục chứa ảnh: ").strip()
    
    if not os.path.exists(folder_path):
        print("❌ Thư mục không tồn tại!")
        return
    
    # Lấy tất cả file ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print("❌ Không tìm thấy ảnh nào trong thư mục!")
        return
    
    print(f"\n✓ Tìm thấy {len(image_files)} ảnh")
    print("🔍 Đang phân loại...\n")
    
    # Phân loại
    results = classifier.predict_batch(image_files)
    
    # Hiển thị kết quả
    for item in results:
        img_name = os.path.basename(item['image'])
        result = item['result']
        print(f"📄 {img_name:30s} → {result['class_name_vi']:15s} ({result['confidence']:.1f}%)")
    
    # Thống kê
    classifier.get_statistics(results)
    
    # Lưu kết quả
    save = input("\nLưu kết quả ra file CSV? (y/n): ").strip().lower()
    if save == 'y':
        import csv
        csv_path = "batch_results.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['File', 'Class', 'Class_VI', 'Confidence'])
            for item in results:
                result = item['result']
                writer.writerow([
                    os.path.basename(item['image']),
                    result['class'],
                    result['class_name_vi'],
                    f"{result['confidence']:.2f}"
                ])
        print(f"✅ Đã lưu kết quả tại: {csv_path}")


def train_new_model():
    """Huấn luyện model mới"""
    print("\n🎓 HUẤN LUYỆN MODEL MỚI")
    print("-" * 70)
    
    print("\n📚 Cấu trúc thư mục training data:")
    print("""
    dataset/
    ├── train/
    │   ├── plastic/
    │   ├── paper/
    │   ├── glass/
    │   ├── metal/
    │   ├── cardboard/
    │   └── trash/
    └── validation/
        ├── plastic/
        ├── paper/
        ├── glass/
        ├── metal/
        ├── cardboard/
        └── trash/
    """)
    
    train_dir = input("\n📁 Đường dẫn thư mục training: ").strip()
    val_dir = input("📁 Đường dẫn thư mục validation: ").strip()
    
    if not os.path.exists(train_dir):
        print(f"❌ Không tìm thấy: {train_dir}")
        return
    
    if not os.path.exists(val_dir):
        print(f"❌ Không tìm thấy: {val_dir}")
        return
    
    # Cấu hình training
    use_transfer = input("\n🔄 Sử dụng Transfer Learning? (y/n): ").strip().lower() == 'y'
    
    epochs_input = input(f"⏱️  Số epochs (Enter = 50): ").strip()
    epochs = int(epochs_input) if epochs_input else 50
    
    # Training
    try:
        print("\n🚀 Bắt đầu training...")
        model, history = train_model(
            train_dir,
            val_dir,
            epochs=epochs,
            use_transfer_learning=use_transfer
        )
        
        # Vẽ biểu đồ
        plot_training_history(history)
        
        print("\n✅ TRAINING HOÀN TẤT!")
        print(f"Model đã lưu tại: {PATHS['model_save']}")
        
    except Exception as e:
        print(f"\n❌ Lỗi training: {str(e)}")


def show_model_info():
    """Hiển thị thông tin model"""
    print("\n📊 THÔNG TIN MODEL")
    print("-" * 70)
    
    model_path = PATHS['model_save']
    if not os.path.exists(model_path):
        model_path = PATHS['best_model']
    
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model!")
        return
    
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    
    print(f"\n📂 Model path: {model_path}")
    print(f"📦 Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"🏷️  Classes: {', '.join(CLASSES)}")
    print(f"📐 Input shape: {model.input_shape}")
    print(f"📊 Output shape: {model.output_shape}")
    
    print("\n" + "="*70)
    print("KIẾN TRÚC MODEL:")
    print("="*70)
    model.summary()
    
    total_params = model.count_params()
    print(f"\n✓ Tổng số parameters: {total_params:,}")


def show_guide():
    """Hiển thị hướng dẫn"""
    guide = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    📖 HƯỚNG DẪN SỬ DỤNG                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    🎯 BƯỚC 1: CHUẨN BỊ DỮ LIỆU
    -------------------------
    - Tải dataset từ Kaggle hoặc tự thu thập
    - Tổ chức thư mục theo cấu trúc:
      dataset/train/[plastic, paper, glass, metal, cardboard, trash]/
      dataset/validation/[plastic, paper, glass, metal, cardboard, trash]/
    
    📚 NGUỒN DATASET:
    - Kaggle: "Waste Classification Data"
    - Kaggle: "TrashNet Dataset"
    
    🎓 BƯỚC 2: HUẤN LUYỆN MODEL
    -------------------------
    - Chọn option 5 trong menu
    - Nhập đường dẫn thư mục train và validation
    - Chọn số epochs (khuyến nghị: 50-100)
    - Đợi training hoàn tất
    
    🔍 BƯỚC 3: SỬ DỤNG
    -------------------------
    - Option 1: Phân loại ảnh đơn lẻ
    - Option 2: Phân loại real-time từ camera
    - Option 3: Phân loại video
    - Option 4: Phân loại hàng loạt ảnh
    
    💡 TIPS:
    -------------------------
    - Dùng Transfer Learning nếu dataset nhỏ (<5000 ảnh)
    - Tăng epochs nếu muốn accuracy cao hơn
    - Test model với nhiều loại ảnh khác nhau
    
    ⚙️ CÀI ĐẶT THƯ VIỆN:
    -------------------------
    pip install tensorflow opencv-python pillow matplotlib numpy
    
    📧 YÊU CẦU HỆ THỐNG:
    -------------------------
    - Python 3.7+
    - TensorFlow 2.x
    - OpenCV
    - Camera (cho real-time classification)
    """
    print(guide)


def main():
    """Main function"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n👉 Nhập lựa chọn: ").strip()
            
            if choice == '1':
                classify_from_image()
            
            elif choice == '2':
                classify_from_camera()
            
            elif choice == '3':
                classify_from_video()
            
            elif choice == '4':
                classify_batch()
            
            elif choice == '5':
                train_new_model()
            
            elif choice == '6':
                show_model_info()
            
            elif choice == '7':
                show_guide()
            
            elif choice == '0':
                print("\n👋 Cảm ơn đã sử dụng! Tạm biệt!\n")
                sys.exit(0)
            
            else:
                print("\n❌ Lựa chọn không hợp lệ!")
            
            input("\n⏸️  Nhấn Enter để tiếp tục...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!\n")
            sys.exit(0)
        
        except Exception as e:
            print(f"\n❌ Lỗi: {str(e)}")
            input("\n⏸️  Nhấn Enter để tiếp tục...")


if __name__ == "__main__":
    main()