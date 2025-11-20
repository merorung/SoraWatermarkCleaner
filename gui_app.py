"""
Simple GUI Application for Sora Watermark Cleaner
Uses tkinter for a lightweight desktop interface
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import threading
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk

from sorawm.core import SoraWM, IMAGE_EXTENSIONS
from sorawm.schemas import CleanerType


class WatermarkCleanerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("소라 워터마크 제거기 v2.0 (이미지+비디오)")
        self.root.geometry("650x650")
        self.root.resizable(False, False)

        # Variables
        self.input_path = None
        self.output_path = None
        self.sora_wm = None
        self.processing = False
        self.cancel_requested = False  # Flag for cancellation
        self.has_gpu = torch.cuda.is_available()
        self.manual_bbox = None  # (x1, y1, x2, y2) for manual selection
        self.app_state = "READY"  # States: READY, PROCESSING, COMPLETED
        self.temp_output_path = None  # Store temp path for completed processing

        self.setup_ui()

    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="🎬 소라 워터마크 제거기",
            font=("맑은 고딕", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)

        # Main content
        content_frame = tk.Frame(self.root, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # GPU status
        gpu_frame = tk.Frame(content_frame)
        gpu_frame.pack(fill=tk.X, pady=(0, 10))

        gpu_status = "✅ GPU 사용 가능" if self.has_gpu else "⚠️ CPU 모드 (느림)"
        gpu_color = "green" if self.has_gpu else "orange"

        tk.Label(
            gpu_frame,
            text=gpu_status,
            font=("맑은 고딕", 9),
            fg=gpu_color
        ).pack(anchor=tk.W)

        # Model selection
        model_frame = tk.LabelFrame(content_frame, text="모델 선택", padx=10, pady=10, font=("맑은 고딕", 9, "bold"))
        model_frame.pack(fill=tk.X, pady=(0, 15))

        self.model_var = tk.StringVar(value=CleanerType.LAMA)

        lama_radio = tk.Radiobutton(
            model_frame,
            text="🚀 LAMA (빠름, 기본 품질)",
            variable=self.model_var,
            value=CleanerType.LAMA,
            font=("맑은 고딕", 10)
        )
        lama_radio.pack(anchor=tk.W)

        # MAT option
        mat_radio = tk.Radiobutton(
            model_frame,
            text="⭐ MAT (빠름, LAMA보다 좋은 품질)",
            variable=self.model_var,
            value=CleanerType.MAT,
            font=("맑은 고딕", 10)
        )
        mat_radio.pack(anchor=tk.W)

        # E2FGVI option (faster version)
        e2fgvi_text = "🌟 E2FGVI (좋은 품질, 시간 일관성, HQ보다 2-3배 빠름)"
        if not self.has_gpu:
            e2fgvi_text += " ⚠️ GPU 권장"

        e2fgvi_std_radio = tk.Radiobutton(
            model_frame,
            text=e2fgvi_text,
            variable=self.model_var,
            value=CleanerType.E2FGVI,
            font=("맑은 고딕", 10),
            fg="gray" if not self.has_gpu else "black"
        )
        e2fgvi_std_radio.pack(anchor=tk.W)

        # E2FGVI-HQ option with GPU warning
        e2fgvi_hq_text = "💎 E2FGVI-HQ (최고 품질, 완벽한 시간 일관성, 가장 느림)"
        if not self.has_gpu:
            e2fgvi_hq_text += " ⚠️ GPU 필요 - CPU에서는 매우 느림"

        e2fgvi_hq_radio = tk.Radiobutton(
            model_frame,
            text=e2fgvi_hq_text,
            variable=self.model_var,
            value=CleanerType.E2FGVI_HQ,
            font=("맑은 고딕", 10),
            fg="gray" if not self.has_gpu else "black"
        )
        e2fgvi_hq_radio.pack(anchor=tk.W)

        # Detection mode selection
        detection_frame = tk.LabelFrame(content_frame, text="워터마크 감지 방식", padx=10, pady=10, font=("맑은 고딕", 9, "bold"))
        detection_frame.pack(fill=tk.X, pady=(0, 15))

        self.detection_mode = tk.StringVar(value="auto")

        auto_radio = tk.Radiobutton(
            detection_frame,
            text="🤖 자동 감지 (AI가 워터마크 자동 찾기)",
            variable=self.detection_mode,
            value="auto",
            font=("맑은 고딕", 10)
        )
        auto_radio.pack(anchor=tk.W)

        manual_radio = tk.Radiobutton(
            detection_frame,
            text="✋ 수동 선택 (직접 워터마크 영역 지정)",
            variable=self.detection_mode,
            value="manual",
            font=("맑은 고딕", 10)
        )
        manual_radio.pack(anchor=tk.W)

        # Manual selection button
        self.manual_select_btn = tk.Button(
            detection_frame,
            text="📍 워터마크 영역 선택하기",
            command=self.select_watermark_area,
            font=("맑은 고딕", 9),
            state=tk.DISABLED
        )
        self.manual_select_btn.pack(anchor=tk.W, padx=20, pady=(5, 0))

        # Status label for manual selection
        self.manual_status_label = tk.Label(
            detection_frame,
            text="",
            font=("맑은 고딕", 8),
            fg="gray"
        )
        self.manual_status_label.pack(anchor=tk.W, padx=20)

        # Input file selection
        input_frame = tk.Frame(content_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(input_frame, text="입력 파일:", font=("맑은 고딕", 10, "bold")).pack(anchor=tk.W)

        input_path_frame = tk.Frame(input_frame)
        input_path_frame.pack(fill=tk.X, pady=(5, 0))

        self.input_label = tk.Label(
            input_path_frame,
            text="파일을 선택하세요",
            font=("맑은 고딕", 9),
            fg="gray",
            anchor=tk.W,
            width=50
        )
        self.input_label.pack(side=tk.LEFT, padx=(0, 10))

        input_btn = tk.Button(
            input_path_frame,
            text="파일 선택...",
            command=self.select_input_file,
            width=12,
            font=("맑은 고딕", 9)
        )
        input_btn.pack(side=tk.RIGHT)

        # Note: Output file selection removed - will be done after processing

        # Progress section
        progress_frame = tk.Frame(content_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 15))

        self.progress_label = tk.Label(
            progress_frame,
            text="준비 완료",
            font=("맑은 고딕", 9),
            fg="green"
        )
        self.progress_label.pack(anchor=tk.W, pady=(0, 5))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=610
        )
        self.progress_bar.pack(fill=tk.X)

        # Main action button (changes based on state)
        self.action_btn = tk.Button(
            content_frame,
            text="🚀 워터마크 제거 시작",
            command=self.handle_action_button,
            font=("맑은 고딕", 14, "bold"),
            bg="#27ae60",
            fg="white",
            height=2,
            cursor="hand2"
        )
        self.action_btn.pack(fill=tk.X)

    def select_input_file(self):
        filename = filedialog.askopenfilename(
            title="입력 파일 선택 (비디오 또는 이미지)",
            filetypes=[
                ("모든 미디어 파일", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"),
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                ("이미지 파일", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"),
                ("모든 파일", "*.*")
            ]
        )
        if filename:
            self.input_path = Path(filename)
            self.input_label.config(
                text=self.input_path.name,
                fg="black"
            )
            # Enable manual selection button when video is loaded
            self.manual_select_btn.config(state=tk.NORMAL)
            # Reset manual bbox when new video is selected
            self.manual_bbox = None
            self.manual_status_label.config(text="")

            # Reset state when new video is selected
            self.app_state = "READY"
            self.update_action_button()
            self.output_path = None
            self.temp_output_path = None

    def update_action_button(self):
        """Update the action button appearance based on current state"""
        if self.app_state == "READY":
            self.action_btn.config(
                text="🚀 워터마크 제거 시작",
                bg="#27ae60",
                fg="white",
                state=tk.NORMAL
            )
        elif self.app_state == "PROCESSING":
            self.action_btn.config(
                text="⏹ 취소",
                bg="#e74c3c",
                fg="white",
                state=tk.NORMAL
            )
        elif self.app_state == "COMPLETED":
            self.action_btn.config(
                text="💾 저장하기",
                bg="#3498db",
                fg="white",
                state=tk.NORMAL
            )

    def handle_action_button(self):
        """Handle action button click based on current state"""
        if self.app_state == "READY":
            self.start_processing()
        elif self.app_state == "PROCESSING":
            self.cancel_processing()
        elif self.app_state == "COMPLETED":
            self.save_processed_file()

    def start_processing(self):
        """Start video processing"""
        # Validation
        if not self.input_path:
            messagebox.showwarning("경고", "입력 파일을 선택하세요.")
            return

        if not self.input_path.exists():
            messagebox.showerror("오류", "입력 파일이 존재하지 않습니다.")
            return

        if self.processing:
            messagebox.showinfo("알림", "이미 처리 중입니다.")
            return

        # Check if manual mode is selected but no area specified
        if self.detection_mode.get() == "manual" and not self.manual_bbox:
            messagebox.showwarning("경고", "수동 선택 모드에서는 워터마크 영역을 먼저 선택해야 합니다.")
            return

        # Warn if using E2FGVI models on CPU
        if self.model_var.get() in [CleanerType.E2FGVI, CleanerType.E2FGVI_HQ] and not self.has_gpu:
            model_name = "E2FGVI-HQ" if self.model_var.get() == CleanerType.E2FGVI_HQ else "E2FGVI"
            result = messagebox.askyesno(
                "경고",
                f"{model_name} 모델은 CPU에서 매우 느립니다.\n"
                "처리 시간이 매우 오래 걸릴 수 있습니다.\n\n"
                "계속하시겠습니까?"
            )
            if not result:
                return

        # Create temporary output path
        self.temp_output_path = self.input_path.parent / f"temp_processed_{self.input_path.name}"

        # Start processing in a separate thread
        self.processing = True
        self.cancel_requested = False
        self.app_state = "PROCESSING"
        self.update_action_button()
        self.progress_bar['value'] = 0

        thread = threading.Thread(target=self.process_video_thread, daemon=True)
        thread.start()

    def save_processed_file(self):
        """Save the processed file to user-selected location"""
        if not self.temp_output_path or not self.temp_output_path.exists():
            messagebox.showerror("오류", "처리된 파일을 찾을 수 없습니다.")
            self.app_state = "READY"
            self.update_action_button()
            return

        # Open save dialog
        initial_name = f"cleaned_{self.input_path.name}" if self.input_path else "output.mp4"
        initial_dir = self.input_path.parent if self.input_path else None

        filename = filedialog.asksaveasfilename(
            title="처리된 비디오 저장",
            initialfile=initial_name,
            initialdir=initial_dir,
            defaultextension=".mp4",
            filetypes=[
                ("MP4 파일", "*.mp4"),
                ("AVI 파일", "*.avi"),
                ("모든 파일", "*.*")
            ]
        )

        if filename:
            try:
                final_path = Path(filename)
                # Move temp file to final location
                import shutil
                shutil.move(str(self.temp_output_path), str(final_path))

                file_size = final_path.stat().st_size / (1024 * 1024)  # MB
                messagebox.showinfo(
                    "저장 완료",
                    f"파일이 저장되었습니다!\n\n"
                    f"파일: {final_path.name}\n"
                    f"크기: {file_size:.2f} MB\n"
                    f"위치: {final_path.parent}"
                )

                # Reset to ready state
                self.app_state = "READY"
                self.update_action_button()
                self.progress_bar['value'] = 0
                self.progress_label.config(text="준비 완료", fg="green")
                self.temp_output_path = None

            except Exception as e:
                messagebox.showerror("저장 오류", f"파일 저장 중 오류가 발생했습니다:\n\n{str(e)}")
        else:
            # User cancelled save - temp file still exists, keep in COMPLETED state
            pass

    def select_output_file(self):
        initial_name = f"cleaned_{self.input_path.name}" if self.input_path else "output.mp4"
        initial_dir = self.input_path.parent if self.input_path else None

        filename = filedialog.asksaveasfilename(
            title="출력 비디오 저장 위치",
            initialfile=initial_name,
            initialdir=initial_dir,
            defaultextension=".mp4",
            filetypes=[
                ("MP4 파일", "*.mp4"),
                ("AVI 파일", "*.avi"),
                ("모든 파일", "*.*")
            ]
        )
        if filename:
            self.output_path = Path(filename)
            self.output_label.config(
                text=self.output_path.name,
                fg="black"
            )

    def select_watermark_area(self):
        """Open window to manually select watermark area with video/image preview"""
        if not self.input_path or not self.input_path.exists():
            messagebox.showwarning("경고", "먼저 파일을 선택하세요.")
            return

        # Check if input is image or video
        is_image = self.input_path.suffix.lower() in IMAGE_EXTENSIONS

        if is_image:
            # Load image
            frame = cv2.imread(str(self.input_path))
            if frame is None:
                messagebox.showerror("오류", "이미지를 열 수 없습니다.")
                return
            total_frames = 1
            fps = 0
            cap = None
        else:
            # Open video
            cap = cv2.VideoCapture(str(self.input_path))
            if not cap.isOpened():
                messagebox.showerror("오류", "비디오를 열 수 없습니다.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Read first frame
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("오류", "비디오 첫 프레임을 읽을 수 없습니다.")
                cap.release()
                return

        # Create selection window
        self.selection_window = tk.Toplevel(self.root)
        self.selection_window.title("워터마크 영역 선택")
        self.selection_window.grab_set()

        # Instructions
        instruction_frame = tk.Frame(self.selection_window)
        instruction_frame.pack(pady=10)

        instruction_label = tk.Label(
            instruction_frame,
            text="마우스로 드래그하여 워터마크 영역을 선택하세요 (여러 영역 선택 가능)",
            font=("맑은 고딕", 10, "bold")
        )
        instruction_label.pack()

        # Video/Image info
        if is_image:
            info_text = "이미지 파일"
        else:
            info_text = f"총 {total_frames} 프레임, {fps:.1f} FPS"

        info_label = tk.Label(
            instruction_frame,
            text=info_text,
            font=("맑은 고딕", 9),
            fg="gray"
        )
        info_label.pack()

        # Get frame dimensions
        h, w = frame.shape[:2]

        # Scale down if too large
        max_display_width = 900
        max_display_height = 600
        scale = min(max_display_width / w, max_display_height / h, 1.0)
        display_w = int(w * scale)
        display_h = int(h * scale)

        self.selection_scale = scale
        self.original_frame_size = (w, h)
        self.video_cap = cap
        self.current_frame_idx = 0
        self.total_frames = total_frames
        self.is_playing = False
        self.selected_bboxes = []  # List of multiple bboxes

        # Canvas for video display
        canvas = tk.Canvas(
            self.selection_window,
            width=display_w,
            height=display_h,
            cursor="cross",
            bg="black"
        )
        canvas.pack(padx=10, pady=10)
        self.canvas = canvas

        # Display first frame
        self.display_frame(frame)

        # Selection variables
        self.sel_start_x = None
        self.sel_start_y = None
        self.current_rect = None

        def on_mouse_down(event):
            self.sel_start_x = event.x
            self.sel_start_y = event.y

        def on_mouse_drag(event):
            if self.sel_start_x is not None:
                if self.current_rect:
                    canvas.delete(self.current_rect)
                self.current_rect = canvas.create_rectangle(
                    self.sel_start_x, self.sel_start_y,
                    event.x, event.y,
                    outline="red", width=2
                )

        def on_mouse_up(event):
            if self.sel_start_x is not None:
                # Calculate bbox in original frame coordinates
                x1 = int(min(self.sel_start_x, event.x) / self.selection_scale)
                y1 = int(min(self.sel_start_y, event.y) / self.selection_scale)
                x2 = int(max(self.sel_start_x, event.x) / self.selection_scale)
                y2 = int(max(self.sel_start_y, event.y) / self.selection_scale)

                # Ensure bbox is within frame
                x1 = max(0, min(x1, self.original_frame_size[0]))
                y1 = max(0, min(y1, self.original_frame_size[1]))
                x2 = max(0, min(x2, self.original_frame_size[0]))
                y2 = max(0, min(y2, self.original_frame_size[1]))

                if x2 - x1 > 5 and y2 - y1 > 5:  # Minimum size
                    # Add to selected bboxes list
                    self.selected_bboxes.append((x1, y1, x2, y2))
                    # Keep the rectangle on canvas (don't delete it)
                    canvas.itemconfig(self.current_rect, outline="green", width=2)
                    update_bbox_list()
                    self.current_rect = None
                else:
                    if self.current_rect:
                        canvas.delete(self.current_rect)
                        self.current_rect = None

                self.sel_start_x = None
                self.sel_start_y = None

        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)

        # Control frame (only for videos)
        if not is_image:
            control_frame = tk.Frame(self.selection_window)
            control_frame.pack(fill=tk.X, padx=10, pady=5)

            # Frame slider
            slider_frame = tk.Frame(control_frame)
            slider_frame.pack(fill=tk.X, pady=(0, 5))

            self.frame_slider = tk.Scale(
                slider_frame,
                from_=0,
                to=total_frames - 1,
                orient=tk.HORIZONTAL,
                command=self.on_slider_change,
                length=display_w - 100
            )
            self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.frame_label = tk.Label(
                slider_frame,
                text=f"0 / {total_frames}",
                font=("맑은 고딕", 9),
                width=15
            )
            self.frame_label.pack(side=tk.RIGHT)

            # Playback controls
            playback_frame = tk.Frame(control_frame)
            playback_frame.pack()

            self.play_btn = tk.Button(
                playback_frame,
                text="▶ 재생",
                command=self.toggle_play,
                font=("맑은 고딕", 9),
                width=10
            )
            self.play_btn.pack(side=tk.LEFT, padx=5)

            prev_btn = tk.Button(
                playback_frame,
                text="◀ 이전",
                command=lambda: self.seek_frame(-10),
                font=("맑은 고딕", 9),
                width=10
            )
            prev_btn.pack(side=tk.LEFT, padx=5)

            next_btn = tk.Button(
                playback_frame,
                text="다음 ▶",
                command=lambda: self.seek_frame(10),
                font=("맑은 고딕", 9),
                width=10
            )
            next_btn.pack(side=tk.LEFT, padx=5)

        # Selected bboxes list
        bbox_frame = tk.LabelFrame(
            self.selection_window,
            text="선택된 영역",
            font=("맑은 고딕", 9, "bold"),
            padx=10,
            pady=5
        )
        bbox_frame.pack(fill=tk.X, padx=10, pady=5)

        self.bbox_listbox = tk.Listbox(
            bbox_frame,
            height=3,
            font=("맑은 고딕", 9)
        )
        self.bbox_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bbox_btn_frame = tk.Frame(bbox_frame)
        bbox_btn_frame.pack(side=tk.RIGHT, padx=(5, 0))

        delete_btn = tk.Button(
            bbox_btn_frame,
            text="삭제",
            command=self.delete_selected_bbox,
            font=("맑은 고딕", 8),
            width=8
        )
        delete_btn.pack(pady=2)

        clear_btn = tk.Button(
            bbox_btn_frame,
            text="전체 삭제",
            command=self.clear_all_bboxes,
            font=("맑은 고딕", 8),
            width=8
        )
        clear_btn.pack(pady=2)

        def update_bbox_list():
            self.bbox_listbox.delete(0, tk.END)
            for i, (x1, y1, x2, y2) in enumerate(self.selected_bboxes):
                self.bbox_listbox.insert(
                    tk.END,
                    f"영역 {i+1}: ({x2-x1}×{y2-y1}) at ({x1}, {y1})"
                )

        # Bottom buttons
        bottom_frame = tk.Frame(self.selection_window)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)

        confirm_btn = tk.Button(
            bottom_frame,
            text="✓ 확인",
            command=self.confirm_selection,
            font=("맑은 고딕", 10, "bold"),
            bg="#27ae60",
            fg="white",
            width=15
        )
        confirm_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = tk.Button(
            bottom_frame,
            text="✕ 취소",
            command=self.cancel_selection,
            font=("맑은 고딕", 10),
            width=15
        )
        cancel_btn.pack(side=tk.RIGHT, padx=5)

    def display_frame(self, frame):
        """Display a frame on the canvas"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb,
            (int(self.original_frame_size[0] * self.selection_scale),
             int(self.original_frame_size[1] * self.selection_scale))
        )
        img = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("frame")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags="frame")

        # Redraw all selected bboxes
        self.canvas.delete("bbox")
        for x1, y1, x2, y2 in self.selected_bboxes:
            self.canvas.create_rectangle(
                x1 * self.selection_scale,
                y1 * self.selection_scale,
                x2 * self.selection_scale,
                y2 * self.selection_scale,
                outline="green",
                width=2,
                tags="bbox"
            )

    def on_slider_change(self, value):
        """Handle slider change"""
        frame_idx = int(value)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame_idx = frame_idx
            self.display_frame(frame)
            self.frame_label.config(text=f"{frame_idx} / {self.total_frames}")

    def toggle_play(self):
        """Toggle video playback"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="⏸ 일시정지")
            self.play_video()
        else:
            self.play_btn.config(text="▶ 재생")

    def play_video(self):
        """Play video"""
        if not self.is_playing:
            return

        if self.current_frame_idx >= self.total_frames - 1:
            self.current_frame_idx = 0
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame_idx += 1
            self.display_frame(frame)
            self.frame_slider.set(self.current_frame_idx)
            self.frame_label.config(text=f"{self.current_frame_idx} / {self.total_frames}")

            # Schedule next frame (30 FPS for smooth playback)
            self.selection_window.after(33, self.play_video)
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ 재생")

    def seek_frame(self, offset):
        """Seek forward or backward by offset frames"""
        new_idx = max(0, min(self.current_frame_idx + offset, self.total_frames - 1))
        self.frame_slider.set(new_idx)

    def delete_selected_bbox(self):
        """Delete selected bbox from list"""
        selection = self.bbox_listbox.curselection()
        if selection:
            idx = selection[0]
            del self.selected_bboxes[idx]
            self.bbox_listbox.delete(idx)
            # Redraw current frame
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.video_cap.read()
            if ret:
                self.display_frame(frame)

    def clear_all_bboxes(self):
        """Clear all selected bboxes"""
        self.selected_bboxes = []
        self.bbox_listbox.delete(0, tk.END)
        # Redraw current frame
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.video_cap.read()
        if ret:
            self.display_frame(frame)

    def confirm_selection(self):
        """Confirm bbox selection"""
        if not self.selected_bboxes:
            messagebox.showwarning("경고", "최소 하나 이상의 영역을 선택해야 합니다.")
            return

        # Store multiple bboxes
        self.manual_bbox = self.selected_bboxes.copy()

        # Update status label
        if len(self.selected_bboxes) == 1:
            x1, y1, x2, y2 = self.selected_bboxes[0]
            self.manual_status_label.config(
                text=f"✓ 선택됨: 1개 영역 ({x2-x1}×{y2-y1} 픽셀)",
                fg="green"
            )
        else:
            self.manual_status_label.config(
                text=f"✓ 선택됨: {len(self.selected_bboxes)}개 영역",
                fg="green"
            )

        self.cancel_selection()

    def cancel_selection(self):
        """Cancel and close selection window"""
        self.is_playing = False
        if hasattr(self, 'video_cap'):
            self.video_cap.release()
        if hasattr(self, 'selection_window'):
            self.selection_window.destroy()

    def cancel_processing(self):
        """Request cancellation of processing"""
        result = messagebox.askyesno(
            "취소 확인",
            "정말로 작업을 취소하시겠습니까?\n진행 중인 작업이 중단됩니다."
        )
        if result:
            self.cancel_requested = True
            self.progress_label.config(text="⏹ 취소 중...", fg="orange")

    def update_progress(self, progress: int):
        """Callback function for progress updates"""
        # Check for cancellation
        if self.cancel_requested:
            return False  # Signal to stop processing

        self.progress_bar['value'] = progress

        if progress < 50:
            status = f"🔍 워터마크 감지 중... {progress}%"
        elif progress < 95:
            status = f"🧹 워터마크 제거 중... {progress}%"
        else:
            status = f"🎵 오디오 병합 중... {progress}%"

        self.progress_label.config(text=status, fg="blue")
        self.root.update_idletasks()
        return True  # Continue processing

    def process_video_thread(self):
        """Process video or image in a separate thread"""
        try:
            # Initialize model if not already done
            if not self.sora_wm or self.sora_wm.cleaner_type != self.model_var.get():
                self.progress_label.config(text="모델 로딩 중...", fg="blue")
                self.root.update_idletasks()
                self.sora_wm = SoraWM(cleaner_type=CleanerType(self.model_var.get()))

            # Check cancellation before starting
            if self.cancel_requested:
                raise InterruptedError("작업이 취소되었습니다.")

            # Check if input is image or video
            is_image = self.input_path.suffix.lower() in IMAGE_EXTENSIONS

            # Process the file to temp location
            if is_image:
                # Image processing
                if self.detection_mode.get() == "manual" and self.manual_bbox:
                    self.sora_wm.run_image(
                        self.input_path,
                        self.temp_output_path,
                        progress_callback=self.update_progress,
                        manual_bbox=self.manual_bbox
                    )
                else:
                    self.sora_wm.run_image(
                        self.input_path,
                        self.temp_output_path,
                        progress_callback=self.update_progress
                    )
            else:
                # Video processing
                if self.detection_mode.get() == "manual" and self.manual_bbox:
                    self.sora_wm.run(
                        self.input_path,
                        self.temp_output_path,
                        progress_callback=self.update_progress,
                        manual_bbox=self.manual_bbox
                    )
                else:
                    self.sora_wm.run(
                        self.input_path,
                        self.temp_output_path,
                        progress_callback=self.update_progress
                    )

            # Check if cancelled during processing
            if self.cancel_requested:
                raise InterruptedError("작업이 취소되었습니다.")

            # Success - verify temp file exists
            if self.temp_output_path.exists():
                file_size = self.temp_output_path.stat().st_size / (1024 * 1024)  # MB
                self.progress_bar['value'] = 100
                self.progress_label.config(text=f"✅ 처리 완료! ({file_size:.1f} MB)", fg="green")

                # Change to COMPLETED state
                self.app_state = "COMPLETED"
                self.update_action_button()
            else:
                raise RuntimeError("처리된 파일을 찾을 수 없습니다.")

        except InterruptedError as e:
            # Clean up temp file on cancellation
            if self.temp_output_path and self.temp_output_path.exists():
                try:
                    self.temp_output_path.unlink()
                except:
                    pass
            # Also clean up the temp file created by core.py
            if self.temp_output_path:
                core_temp = self.temp_output_path.parent / f"temp_{self.temp_output_path.name}"
                if core_temp.exists():
                    try:
                        core_temp.unlink()
                    except:
                        pass

            self.progress_bar['value'] = 0
            self.progress_label.config(text="⏹ 작업이 취소되었습니다.", fg="orange")
            self.app_state = "READY"
            self.update_action_button()
            messagebox.showinfo("취소됨", str(e))

        except Exception as e:
            self.progress_label.config(text=f"❌ 오류: {str(e)}", fg="red")
            self.app_state = "READY"
            self.update_action_button()
            messagebox.showerror("오류", f"오류가 발생했습니다:\n\n{str(e)}")

            # Clean up temp file on error
            if self.temp_output_path and self.temp_output_path.exists():
                try:
                    self.temp_output_path.unlink()
                except:
                    pass

        finally:
            self.processing = False
            self.cancel_requested = False



def main():
    root = tk.Tk()
    app = WatermarkCleanerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
