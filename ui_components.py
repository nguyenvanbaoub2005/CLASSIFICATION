# ui_components.py
"""
Module chứa các component UI tái sử dụng
"""

import tkinter as tk
from tkinter import ttk


class ModernButton(tk.Button):
    """Custom modern button với shadow effect"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.config(
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            activebackground=kwargs.get('bg', '#0066cc')
        )
        
        # Hover effect
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.default_bg = kwargs.get('bg', '#0066cc')
    
    def on_enter(self, e):
        self['background'] = self.lighten_color(self.default_bg)
    
    def on_leave(self, e):
        self['background'] = self.default_bg
    
    def lighten_color(self, color):
        """Làm sáng màu khi hover"""
        color_map = {
            '#0066cc': '#0077ee',
            '#28a745': '#32d956',
            '#dc3545': '#ff4757',
            '#ffc107': '#ffd43b',
            '#6c757d': '#868e96',
            '#17a2b8': '#1ac9e6',
            '#6f42c1': '#8357d8',
            '#fd7e14': '#ff922b',
        }
        return color_map.get(color, color)


class Card(tk.Frame):
    """Component Card với shadow"""
    
    def __init__(self, parent, title=None, colors=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        if colors is None:
            colors = {
                'card': '#ffffff',
                'text': '#212529',
                'border': '#dee2e6'
            }
        
        self.colors = colors
        self.config(
            bg=self.colors['card'],
            relief='flat',
            bd=0
        )
        
        if title:
            self._create_title(title)
    
    def _create_title(self, title):
        """Tạo title cho card"""
        title_label = tk.Label(
            self,
            text=title,
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        title_label.pack(pady=(15, 10), padx=20, anchor='w')
        
        # Separator line
        separator = tk.Frame(self, height=2, bg=self.colors['border'])
        separator.pack(fill='x', padx=20)


class StatusIndicator(tk.Frame):
    """Component hiển thị status"""
    
    def __init__(self, parent, colors=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        if colors is None:
            colors = {
                'bg': '#ffffff',
                'success': '#28a745',
                'danger': '#dc3545',
                'text': '#212529'
            }
        
        self.colors = colors
        self.config(bg=self.colors['bg'])
        
        # Status dot
        self.status_dot = tk.Label(
            self,
            text="●",
            font=('Arial', 20),
            bg=self.colors['bg']
        )
        self.status_dot.pack(side='left')
        
        # Status text
        self.status_label = tk.Label(
            self,
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg']
        )
        self.status_label.pack(side='left', padx=(5, 0))
    
    def set_status(self, text, is_success=True):
        """Set trạng thái"""
        color = self.colors['success'] if is_success else self.colors['danger']
        self.status_dot.config(fg=color)
        self.status_label.config(text=text, fg=color)


class Header(tk.Frame):
    """Component Header"""
    
    def __init__(self, parent, title, subtitle=None, colors=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        if colors is None:
            colors = {
                'header': '#ffffff',
                'primary': '#0066cc',
                'text_secondary': '#6c757d'
            }
        
        self.colors = colors
        self.config(bg=self.colors['header'], height=80)
        self.pack_propagate(False)
        
        # Shadow
        shadow = tk.Frame(parent, height=3, bg='#dee2e6')
        shadow.pack(fill='x')
        
        # Content
        content = tk.Frame(self, bg=self.colors['header'])
        content.pack(fill='both', expand=True, padx=30)
        
        # Title
        title_frame = tk.Frame(content, bg=self.colors['header'])
        title_frame.pack(side='left', pady=20)
        
        title_label = tk.Label(
            title_frame,
            text=title,
            font=('Segoe UI', 26, 'bold'),
            bg=self.colors['header'],
            fg=self.colors['primary']
        )
        title_label.pack(side='left')
        
        if subtitle:
            subtitle_label = tk.Label(
                title_frame,
                text=subtitle,
                font=('Segoe UI', 11),
                bg=self.colors['header'],
                fg=self.colors['text_secondary']
            )
            subtitle_label.pack(side='left', padx=(15, 0))
        
        # Status indicator (placeholder)
        self.status_frame = tk.Frame(content, bg=self.colors['header'])
        self.status_frame.pack(side='right', pady=20)
    
    def add_status_indicator(self, status_widget):
        """Thêm status indicator vào header"""
        status_widget.pack(in_=self.status_frame)


class Sidebar(tk.Frame):
    """Component Sidebar menu"""
    
    def __init__(self, parent, colors=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        if colors is None:
            colors = {
                'card': '#ffffff',
                'text': '#212529'
            }
        
        self.colors = colors
        self.config(
            bg=self.colors['card'],
            width=220
        )
        self.pack_propagate(False)
        
        # Menu title
        menu_title = tk.Label(
            self,
            text="📋 MENU",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        menu_title.pack(pady=(20, 15))
        
        # Buttons container
        self.buttons_container = tk.Frame(self, bg=self.colors['card'])
        self.buttons_container.pack(fill='both', expand=True)
    
    def add_button(self, text, command, color):
        """Thêm button vào sidebar"""
        btn = ModernButton(
            self.buttons_container,
            text=text,
            bg=color,
            fg='white',
            command=command,
            width=16
        )
        btn.pack(pady=6, padx=15)
        return btn
    
    def add_spacer(self):
        """Thêm spacer"""
        tk.Frame(self, bg=self.colors['card']).pack(expand=True)


class VideoPanel(tk.Frame):
    """Component hiển thị video/camera"""
    
    def __init__(self, parent, title, colors=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        if colors is None:
            colors = {
                'card': '#ffffff',
                'border': '#dee2e6'
            }
        
        self.colors = colors
        
        # Card wrapper
        self.card = Card(self, title=title, colors=colors)
        self.card.pack(fill='both', expand=True)
        
        # Auto scan toggle
        self.auto_scan_var = tk.BooleanVar()
        self._create_toggle()
        
        # Video frame
        self._create_video_frame()
        
        # Controls
        self.controls_frame = tk.Frame(self.card, bg=self.colors['card'])
        self.controls_frame.pack(pady=(0, 20))
    
    def _create_toggle(self):
        """Tạo toggle auto scan"""
        toggle_frame = tk.Frame(self.card, bg=self.colors['card'])
        toggle_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        self.auto_check = tk.Checkbutton(
            toggle_frame,
            text="🤖 Tự động quét",
            variable=self.auto_scan_var,
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['card'],
            selectcolor=self.colors['card'],
            activebackground=self.colors['card']
        )
        self.auto_check.pack(side='right')
    
    def _create_video_frame(self):
        """Tạo video frame"""
        video_container = tk.Frame(
            self.card,
            bg=self.colors['border'],
            relief='flat',
            bd=2
        )
        video_container.pack(padx=20, pady=15, fill='both', expand=True)
        
        self.video_label = tk.Label(video_container, bg='#000000')
        self.video_label.pack(fill='both', expand=True, padx=2, pady=2)
    
    def add_control_button(self, text, bg, command, width=13):
        """Thêm nút điều khiển"""
        btn = ModernButton(
            self.controls_frame,
            text=text,
            bg=bg,
            fg='white',
            width=width,
            command=command
        )
        btn.pack(side='left', padx=5)
        return btn
    
    def get_video_label(self):
        """Lấy video label"""
        return self.video_label
    
    def get_auto_scan_var(self):
        """Lấy auto scan variable"""
        return self.auto_scan_var


class ResultsPanel(tk.Frame):
    """Component hiển thị kết quả"""
    
    def __init__(self, parent, title, colors=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        if colors is None:
            colors = {
                'card': '#ffffff',
                'border': '#dee2e6',
                'text': '#212529'
            }
        
        self.colors = colors
        self.config(width=520)
        self.pack_propagate(False)
        
        # Card wrapper
        self.card = Card(self, title=title, colors=colors)
        self.card.pack(fill='both', expand=True)
        
        # Result text
        self._create_result_display()
        
        # Action buttons
        self.action_frame = tk.Frame(self.card, bg=self.colors['card'])
        self.action_frame.pack(pady=(0, 15))
        
        # Stats card
        self._create_stats_card()
    
    def _create_result_display(self):
        """Tạo vùng hiển thị kết quả"""
        from tkinter import scrolledtext
        
        result_container = tk.Frame(
            self.card,
            bg=self.colors['border'],
            relief='flat',
            bd=1
        )
        result_container.pack(fill='both', expand=True, padx=20, pady=(10, 15))
        
        self.result_text = scrolledtext.ScrolledText(
            result_container,
            font=('Consolas', 11),
            bg='#f8f9fa',
            fg=self.colors['text'],
            wrap='word',
            relief='flat',
            bd=0,
            state='disabled',
            padx=15,
            pady=15
        )
        self.result_text.pack(fill='both', expand=True, padx=1, pady=1)
        
        # Configure tags
        self.result_text.tag_config('header', font=('Segoe UI', 13, 'bold'))
        self.result_text.tag_config('success', foreground='#28a745')
        self.result_text.tag_config('warning', foreground='#ffc107')
        self.result_text.tag_config('info', foreground='#17a2b8')
        self.result_text.tag_config('bold', font=('Consolas', 11, 'bold'))
    
    def _create_stats_card(self):
        """Tạo stats card"""
        stats_card = tk.Frame(
            self.card,
            bg='#e7f3ff',
            relief='flat',
            bd=0
        )
        stats_card.pack(fill='x', padx=20, pady=(0, 20))
        
        stats_title = tk.Label(
            stats_card,
            text="📈 Thống Kê Nhanh",
            font=('Segoe UI', 12, 'bold'),
            bg='#e7f3ff',
            fg='#0066cc'
        )
        stats_title.pack(pady=(12, 8), padx=15, anchor='w')
        
        self.stats_label = tk.Label(
            stats_card,
            text="Chưa có dữ liệu",
            font=('Segoe UI', 10),
            bg='#e7f3ff',
            justify='left',
            anchor='w'
        )
        self.stats_label.pack(padx=15, pady=(0, 12), anchor='w')
    
    def add_action_button(self, text, bg, command, width=16):
        """Thêm action button"""
        btn = ModernButton(
            self.action_frame,
            text=text,
            bg=bg,
            fg='white',
            width=width,
            command=command
        )
        btn.pack(side='left', padx=5)
        return btn
    
    def get_result_text(self):
        """Lấy result text widget"""
        return self.result_text
    
    def get_stats_label(self):
        """Lấy stats label"""
        return self.stats_label


def configure_ttk_style():
    """Cấu hình TTK style"""
    style = ttk.Style()
    style.theme_use('clam')
    
    # Treeview styling
    style.configure(
        "Treeview",
        background="#ffffff",
        foreground="#212529",
        fieldbackground="#ffffff",
        borderwidth=1,
        relief='solid',
        rowheight=30
    )
    
    style.configure(
        "Treeview.Heading",
        background="#f8f9fa",
        foreground="#0066cc",
        borderwidth=1,
        relief='solid',
        font=('Segoe UI', 10, 'bold')
    )
    
    style.map(
        'Treeview',
        background=[('selected', '#e3f2fd')],
        foreground=[('selected', '#0066cc')]
    )
    
    # Scrollbar styling
    style.configure(
        "Vertical.TScrollbar",
        background="#dee2e6",
        troughcolor="#f8f9fa",
        borderwidth=0,
        arrowsize=15
    )