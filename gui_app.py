"""
Simple GUI Application for Sora Watermark Cleaner
Uses tkinter for a lightweight desktop interface
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import threading
import torch

from sorawm.core import SoraWM
from sorawm.schemas import CleanerType


class WatermarkCleanerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("소라 워터마크 제거기")
        self.root.geometry("650x500")
        self.root.resizable(False, False)

        # Variables
        self.input_path = None
        self.output_path = None
        self.sora_wm = None
        self.processing = False
        self.has_gpu = torch.cuda.is_available()

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
            text="🚀 LAMA (빠름, 좋은 품질)",
            variable=self.model_var,
            value=CleanerType.LAMA,
            font=("맑은 고딕", 10)
        )
        lama_radio.pack(anchor=tk.W)

        # E2FGVI option with GPU warning
        e2fgvi_text = "💎 E2FGVI-HQ (최고 품질, 시간 일관성 보장)"
        if not self.has_gpu:
            e2fgvi_text += " ⚠️ GPU 필요 - CPU에서는 매우 느림"

        e2fgvi_radio = tk.Radiobutton(
            model_frame,
            text=e2fgvi_text,
            variable=self.model_var,
            value=CleanerType.E2FGVI_HQ,
            font=("맑은 고딕", 10),
            fg="gray" if not self.has_gpu else "black"
        )
        e2fgvi_radio.pack(anchor=tk.W)

        # Input file selection
        input_frame = tk.Frame(content_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(input_frame, text="입력 비디오:", font=("맑은 고딕", 10, "bold")).pack(anchor=tk.W)

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

        # Output file selection
        output_frame = tk.Frame(content_frame)
        output_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(output_frame, text="출력 비디오:", font=("맑은 고딕", 10, "bold")).pack(anchor=tk.W)

        output_path_frame = tk.Frame(output_frame)
        output_path_frame.pack(fill=tk.X, pady=(5, 0))

        self.output_label = tk.Label(
            output_path_frame,
            text="파일을 선택하세요",
            font=("맑은 고딕", 9),
            fg="gray",
            anchor=tk.W,
            width=50
        )
        self.output_label.pack(side=tk.LEFT, padx=(0, 10))

        output_btn = tk.Button(
            output_path_frame,
            text="저장 위치...",
            command=self.select_output_file,
            width=12,
            font=("맑은 고딕", 9)
        )
        output_btn.pack(side=tk.RIGHT)

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

        # Process button
        self.process_btn = tk.Button(
            content_frame,
            text="🚀 워터마크 제거하기",
            command=self.process_video,
            font=("맑은 고딕", 12, "bold"),
            bg="#27ae60",
            fg="white",
            height=2,
            cursor="hand2"
        )
        self.process_btn.pack(fill=tk.X)

    def select_input_file(self):
        filename = filedialog.askopenfilename(
            title="입력 비디오 선택",
            filetypes=[
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv"),
                ("모든 파일", "*.*")
            ]
        )
        if filename:
            self.input_path = Path(filename)
            self.input_label.config(
                text=self.input_path.name,
                fg="black"
            )

            # Auto-suggest output filename
            if not self.output_path:
                output_name = f"cleaned_{self.input_path.name}"
                suggested_output = self.input_path.parent / output_name
                self.output_path = suggested_output
                self.output_label.config(
                    text=output_name,
                    fg="black"
                )

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

    def update_progress(self, progress: int):
        """Callback function for progress updates"""
        self.progress_bar['value'] = progress

        if progress < 50:
            status = f"🔍 워터마크 감지 중... {progress}%"
        elif progress < 95:
            status = f"🧹 워터마크 제거 중... {progress}%"
        else:
            status = f"🎵 오디오 병합 중... {progress}%"

        self.progress_label.config(text=status, fg="blue")
        self.root.update_idletasks()

    def process_video_thread(self):
        """Process video in a separate thread"""
        try:
            # Initialize model if not already done
            if not self.sora_wm or self.sora_wm.cleaner_type != self.model_var.get():
                self.progress_label.config(text="모델 로딩 중...", fg="blue")
                self.root.update_idletasks()
                self.sora_wm = SoraWM(cleaner_type=CleanerType(self.model_var.get()))

            # Process the video
            self.sora_wm.run(
                self.input_path,
                self.output_path,
                progress_callback=self.update_progress
            )

            # Success
            self.progress_bar['value'] = 100
            self.progress_label.config(text="✅ 처리 완료!", fg="green")

            messagebox.showinfo(
                "완료",
                f"워터마크가 성공적으로 제거되었습니다!\n\n저장 위치:\n{self.output_path}"
            )

        except Exception as e:
            self.progress_label.config(text=f"❌ 오류: {str(e)}", fg="red")
            messagebox.showerror("오류", f"오류가 발생했습니다:\n\n{str(e)}")

        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL, bg="#27ae60")

    def process_video(self):
        # Validation
        if not self.input_path:
            messagebox.showwarning("경고", "입력 비디오 파일을 선택하세요.")
            return

        if not self.output_path:
            messagebox.showwarning("경고", "출력 위치를 선택하세요.")
            return

        if not self.input_path.exists():
            messagebox.showerror("오류", "입력 파일이 존재하지 않습니다.")
            return

        if self.processing:
            messagebox.showinfo("알림", "이미 처리 중입니다.")
            return

        # Warn if using E2FGVI on CPU
        if self.model_var.get() == CleanerType.E2FGVI_HQ and not self.has_gpu:
            result = messagebox.askyesno(
                "경고",
                "E2FGVI-HQ 모델은 CPU에서 매우 느립니다.\n"
                "처리 시간이 매우 오래 걸릴 수 있습니다.\n\n"
                "계속하시겠습니까?"
            )
            if not result:
                return

        # Start processing in a separate thread
        self.processing = True
        self.process_btn.config(state=tk.DISABLED, bg="gray")
        self.progress_bar['value'] = 0

        thread = threading.Thread(target=self.process_video_thread, daemon=True)
        thread.start()


def main():
    root = tk.Tk()
    app = WatermarkCleanerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
