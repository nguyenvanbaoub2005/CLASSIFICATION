# 🤖 Hệ Thống Phân Loại Rác Thải Bằng AI

Hệ thống phân loại rác thải tự động sử dụng Deep Learning (CNN) để nhận diện 6 loại rác: **plastic, paper, glass, metal, cardboard, trash**.

## 📋 Mục Lục

- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Sử dụng](#sử-dụng)
- [Huấn luyện model](#huấn-luyện-model)
- [Dataset](#dataset)

## ✨ Tính năng

- ✅ Phân loại 6 loại rác: plastic, paper, glass, metal, cardboard, trash
- ✅ Phân loại từ ảnh đơn lẻ
- ✅ Phân loại real-time từ camera
- ✅ Phân loại từ video file
- ✅ Phân loại batch nhiều ảnh
- ✅ Hiển thị độ tin cậy và hướng dẫn xử lý
- ✅ Hỗ trợ Transfer Learning
- ✅ Data Augmentation tự động

## 💻 Yêu cầu hệ thống

### Phần cứng
- **CPU**: Intel i5 hoặc tương đương
- **RAM**: Tối thiểu 8GB (16GB khuyến nghị)
- **GPU**: NVIDIA GPU với CUDA (không bắt buộc nhưng khuyến nghị cho training)
- **Camera**: Webcam (cho chức năng real-time)

### Phần mềm
- Python 3.7 - 3.10
- pip hoặc conda

## 🔧 Cài đặt

### 1. Clone hoặc tải project

```bash
git clone <repository-url>
cd waste-classifier
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 📁 Cấu trúc dự án

```
waste-classifier/
├── config.py              # Cấu hình hệ thống
├── model.py               # Kiến trúc CNN model
├── train.py               # Huấn luyện model
├── classifier.py          # Class phân loại
├── camera.py              # Xử lý camera/video
├── requirements.txt       # Thư viện cần thiết
├── README.md              # File này
│
├── dataset/               # Thư mục dữ liệu (tự tạo)
│   ├── train/
│   │   ├── plastic/
│   │   ├── paper/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── cardboard/
│   │   └── trash/
│   └── validation/
│       ├── plastic/
│       ├── paper/
│       ├── glass/
│       ├── metal/
│       ├── cardboard/
│       └── trash/
│
└── models/                # Model đã train (tự động tạo)
    ├── waste_classifier_final.h5
    └── waste_classifier_best.h5
```

## 🚀 Sử dụng

### 1. Phân loại từ ảnh

```bash
python classifier.py
```

Nhập đường dẫn ảnh khi được hỏi.

### 2. Phân loại từ camera (Real-time)

```bash
python camera.py
# Chọn option 1
```

Điều khiển:
- **SPACE** - Chụp và phân loại
- **C** - Bật/tắt chế độ liên tục
- **S** - Lưu ảnh
- **Q** - Thoát

### 3. Phân loại từ video

```bash
python camera.py
# Chọn option 2
```

## 🎓 Huấn luyện model

### 1. Chuẩn bị dataset

Tổ chức thư mục theo cấu trúc:

```
dataset/
├── train/
│   ├── plastic/     (500+ ảnh)
│   ├── paper/       (500+ ảnh)
│   ├── glass/       (500+ ảnh)
│   ├── metal/       (500+ ảnh)
│   ├── cardboard/   (500+ ảnh)
│   └── trash/       (500+ ảnh)
└── validation/
    ├── plastic/     (100+ ảnh)
    ├── paper/       (100+ ảnh)
    ├── glass/       (100+ ảnh)
    ├── metal/       (100+ ảnh)
    ├── cardboard/   (100+ ảnh)
    └── trash/       (100+ ảnh)
```

### 2. Chạy training

```bash
python train.py
```

Nhập thông tin khi được hỏi:
- Đường dẫn thư mục train
- Đường dẫn thư mục validation
- Sử dụng Transfer Learning (y/n)
- Số epochs (mặc định: 50)

### 3. Kết quả

Sau khi training xong, bạn sẽ có:
- `waste_classifier_final.h5` - Model cuối cùng
- `waste_classifier_best.h5` - Model tốt nhất
- `training_history.png` - Biểu đồ training

## 📊 Dataset

### Nguồn dataset khuyến nghị:

#### Kaggle
1. **Waste Classification Data**
   - ~25,000 ảnh, 6 classes
   - https://www.kaggle.com/datasets/techsash/waste-classification-data

2. **TrashNet Dataset**
   - ~2,500 ảnh, 6 classes
   - https://www.kaggle.com/datasets/fedesoriano/trashnet

## 📝 Ví dụ sử dụng

### Phân loại một ảnh

```python
from classifier import WasteClassifier

# Khởi tạo
classifier = WasteClassifier('waste_classifier_final.h5')

# Phân loại
result = classifier.predict('test_image.jpg')

# Hiển thị kết quả
classifier.display_result('test_image.jpg', result)
```

### Camera real-time

```python
from camera import CameraClassifier

cam = CameraClassifier('waste_classifier_final.h5')
cam.start_camera()
cam.run_interactive()
```

## 🔧 Cấu hình

Chỉnh sửa trong `config.py`:

```python
# Số epochs
MODEL_CONFIG['epochs'] = 100

# Batch size
MODEL_CONFIG['batch_size'] = 16

# Learning rate
MODEL_CONFIG['learning_rate'] = 0.0001
```

## 🐛 Troubleshooting

### Lỗi ImportError

```bash
pip install --upgrade tensorflow opencv-python pillow matplotlib
```

### Lỗi Out of Memory

Giảm batch_size trong `config.py`:
```python
MODEL_CONFIG['batch_size'] = 8  # hoặc 4
```

### Camera không hoạt động

Thử camera ID khác:
```python
cam.start_camera(camera_id=1)  # hoặc 2, 3
```

## 📈 Kết quả mong đợi

| Metric | Value |
|--------|-------|
| Training Accuracy | 92-95% |
| Validation Accuracy | 85-90% |
| Inference Time | ~100ms/image |
| Model Size | ~50MB |

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra requirements.txt
2. Đảm bảo dataset đúng cấu trúc
3. Xem log lỗi chi tiết

## 📄 License

MIT License - Free to use for educational and commercial purposes.

---

**Happy Coding! 🚀**# CLASSIFICATION
