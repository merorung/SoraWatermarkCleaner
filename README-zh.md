# 图像水印去除工具

[English](README.md) | 中文

这个项目提供了一种优雅的方式，使用AI驱动的图像修复模型来去除图像中的水印。

## 特性

- **AI驱动的水印检测**：使用YOLOv11自动检测水印
- **多种修复模型**：可选择LAMA或MAT模型以获得最佳效果
- **高质量结果**：先进的图像修复技术，实现无缝水印去除
- **易于使用**：简单的Python API和命令行界面

## 方法

图像水印去除工具由两部分组成：

- **水印检测器**：使用训练好的YOLOv11s模型检测图像中的水印
- **水印清除器**：使用先进的修复模型（LAMA或MAT）去除检测到的水印

我们的解决方案完全由深度学习驱动，在各种图像上都能产生出色的结果。

## 安装

### Python环境

我们强烈推荐使用 `uv` 来安装环境：

1. 安装：

```bash
uv sync
```

> 环境将安装在 `.venv` 目录中。您可以使用以下命令激活它：
>
> ```bash
> source .venv/bin/activate
> ```

2. 下载预训练模型：

训练好的YOLO权重将存储在 `resources` 目录中，文件名为 `best.pt`。它将自动从项目发布中下载。LAMA模型从IOPaint项目下载，并将存储在torch缓存目录中。两个下载都是自动的；如果失败，请检查您的网络连接。

## 演示

要进行基本使用，请尝试 `example.py`：

```python
from pathlib import Path

from sorawm.core import SoraWM
from sorawm.schemas import CleanerType

if __name__ == "__main__":
    input_image_path = Path("resources/watermark_template.png")
    output_image_path = Path("outputs/watermark_removed")

    # 1. LAMA快速且提供良好的质量结果
    sora_wm = SoraWM(cleaner_type=CleanerType.LAMA)
    sora_wm.run_image(input_image_path, Path(f"{output_image_path}_lama.png"))

    # 2. MAT是图像修复的另一个选择
    sora_wm = SoraWM(cleaner_type=CleanerType.MAT)
    sora_wm.run_image(input_image_path, Path(f"{output_image_path}_mat.png"))
```

## API使用

### 基本使用

```python
from pathlib import Path
from sorawm.core import SoraWM
from sorawm.schemas import CleanerType

# 使用您喜欢的模型初始化
watermark_remover = SoraWM(cleaner_type=CleanerType.LAMA)

# 从图像中去除水印
watermark_remover.run_image(
    input_image_path=Path("input.jpg"),
    output_image_path=Path("output.jpg")
)
```

### 使用手动边界框的高级用法

```python
# 如果您知道水印位置，可以手动指定
watermark_remover.run_image(
    input_image_path=Path("input.jpg"),
    output_image_path=Path("output.jpg"),
    manual_bbox=(100, 100, 200, 150)  # (x1, y1, x2, y2)
)

# 对于多个水印
watermark_remover.run_image(
    input_image_path=Path("input.jpg"),
    output_image_path=Path("output.jpg"),
    manual_bbox=[(100, 100, 200, 150), (300, 300, 400, 350)]
)
```

### 进度回调

```python
def progress_handler(progress: int):
    print(f"进度: {progress}%")

watermark_remover.run_image(
    input_image_path=Path("input.jpg"),
    output_image_path=Path("output.jpg"),
    progress_callback=progress_handler
)
```

## 可用模型

### LAMA（默认）
- **速度**：快
- **质量**：好
- **最适合**：通用水印去除
- **描述**：大型掩码修复模型，提供出色的结果和快速处理

### MAT
- **速度**：中等
- **质量**：好
- **最适合**：替代修复方法
- **描述**：掩码感知变换器，用于高质量图像修复

## 数据集

标注数据集可在Hugging Face上获得：[sora-watermark-dataset](https://huggingface.co/datasets/LLinked/sora-watermark-dataset)。欢迎训练您的自定义检测器模型或改进我们的模型！

## 许可证

Apache许可证

## 引用

如果您使用此项目，请引用：

```bibtex
@misc{imagewatermarkremover2025,
  author = {linkedlist771},
  title = {Image Watermark Remover},
  year = {2025},
  url = {https://github.com/linkedlist771/SoraWatermarkCleaner}
}
```

## 致谢

- [IOPaint](https://github.com/Sanster/IOPaint) 提供LAMA和MAT实现
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 提供目标检测
