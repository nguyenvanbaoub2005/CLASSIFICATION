# config.py
"""
File cấu hình cho hệ thống phân loại rác thải
"""

# Các loại rác cần phân loại
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'organic']

# Thông tin chi tiết về từng loại rác
CLASS_INFO = {
    'plastic': {
        'name_vi': 'Nhựa',
        'icon': '🥤',
        'color': '\033[94m',  # Blue
        'disposal': 'Tái chế - Rửa sạch và bỏ vào thùng nhựa',
        'examples': ['Chai nước', 'Túi nilon', 'Hộp nhựa', 'Ly nhựa'],
        'recycling_value': 'Cao'
    },
    'paper': {
        'name_vi': 'Giấy',
        'icon': '📄',
        'color': '\033[93m',  # Yellow
        'disposal': 'Tái chế - Bỏ vào thùng giấy',
        'examples': ['Báo cũ', 'Hộp giấy', 'Sách vở', 'Tờ rơi'],
        'recycling_value': 'Trung bình'
    },
    'glass': {
        'name_vi': 'Thủy tinh',
        'icon': '🍾',
        'color': '\033[92m',  # Green
        'disposal': 'Tái chế - Cẩn thận khi xử lý',
        'examples': ['Chai thủy tinh', 'Lọ', 'Cốc', 'Bình'],
        'recycling_value': 'Cao'
    },
    'metal': {
        'name_vi': 'Kim loại',
        'icon': '🥫',
        'color': '\033[90m',  # Gray
        'disposal': 'Tái chế - Bỏ vào thùng kim loại',
        'examples': ['Lon nước ngọt', 'Hộp thiếc', 'Dây kẽm', 'Vỏ lon'],
        'recycling_value': 'Rất cao'
    },
    'cardboard': {
        'name_vi': 'Bìa cứng',
        'icon': '📦',
        'color': '\033[33m',  # Orange
        'disposal': 'Tái chế - Gấp gọn trước khi bỏ',
        'examples': ['Hộp carton', 'Thùng giấy', 'Bìa đóng gói'],
        'recycling_value': 'Trung bình'
    },
    'trash': {
        'name_vi': 'Rác thải thông thường',
        'icon': '🗑️',
        'color': '\033[91m',  # Red
        'disposal': 'Rác thông thường - Bỏ vào thùng rác',
        'examples': ['Rác không tái chế', 'Rác bẩn', 'Rác hữu cơ'],
        'recycling_value': 'Không'
    },
    'organic': {
    'name_vi': 'Rác hữu cơ',
    'icon': '🍌',
    'color': '\033[32m',  # Green
    'disposal': 'Phân hủy – Bỏ vào thùng rác hữu cơ',
    'examples': ['Vỏ trái cây', 'Rau củ', 'Thức ăn thừa'],
    'recycling_value': 'Sinh học'
}

}

# Cấu hình model
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': len(CLASSES),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001
}

# Đường dẫn
PATHS = {
    'model_save': 'waste_classifier_final.h5',
    'best_model': 'waste_classifier_best.h5',
    'temp_image': 'temp_capture.jpg',
    'training_plot': 'training_history.png'
}

# Cấu hình data augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'shear_range': 0.2,
    'fill_mode': 'nearest'
}

# Ngưỡng confidence để cảnh báo
CONFIDENCE_THRESHOLD = 70.0

# Màu sắc terminal
COLORS = {
    'reset': '\033[0m',
    'blue': '\033[94m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'gray': '\033[90m',
    'orange': '\033[33m'
}