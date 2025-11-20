# Image Watermark Remover

English | [中文](README-zh.md)

This project provides an elegant way to remove watermarks from images using AI-powered inpainting models.

## Features

- **AI-Powered Watermark Detection**: Automatically detects watermarks using YOLOv11
- **Multiple Inpainting Models**: Choose between LAMA and MAT models for best results
- **High Quality Results**: Advanced image inpainting for seamless watermark removal
- **Easy to Use**: Simple Python API and command-line interface

## Method

The Image Watermark Remover consists of two parts:

- **WatermarkDetector**: Uses a trained YOLOv11s model to detect watermarks in images
- **WatermarkCleaner**: Uses advanced inpainting models (LAMA or MAT) to remove detected watermarks

Our solution is purely deep learning driven and yields excellent results on various images.

## Installation

### Python Environment

We highly recommend using `uv` to install the environment:

1. Installation:

```bash
uv sync
```

> The environment will be installed at `.venv`. You can activate it using:
>
> ```bash
> source .venv/bin/activate
> ```

2. Download pretrained models:

The trained YOLO weights will be stored in the `resources` directory as `best.pt`. It will be automatically downloaded from the project releases. The LAMA model is downloaded from the IOPaint project and will be stored in the torch cache directory. Both downloads are automatic; if they fail, please check your internet connection.

## Demo

To have a basic usage, try the `example.py`:

```python
from pathlib import Path

from sorawm.core import SoraWM
from sorawm.schemas import CleanerType

if __name__ == "__main__":
    input_image_path = Path("resources/watermark_template.png")
    output_image_path = Path("outputs/watermark_removed")

    # 1. LAMA is fast and provides good quality results
    sora_wm = SoraWM(cleaner_type=CleanerType.LAMA)
    sora_wm.run_image(input_image_path, Path(f"{output_image_path}_lama.png"))

    # 2. MAT is another option for image inpainting
    sora_wm = SoraWM(cleaner_type=CleanerType.MAT)
    sora_wm.run_image(input_image_path, Path(f"{output_image_path}_mat.png"))
```

## API Usage

### Basic Usage

```python
from pathlib import Path
from sorawm.core import SoraWM
from sorawm.schemas import CleanerType

# Initialize with your preferred model
watermark_remover = SoraWM(cleaner_type=CleanerType.LAMA)

# Remove watermark from an image
watermark_remover.run_image(
    input_image_path=Path("input.jpg"),
    output_image_path=Path("output.jpg")
)
```

### Advanced Usage with Manual Bounding Box

```python
# If you know the watermark location, you can specify it manually
watermark_remover.run_image(
    input_image_path=Path("input.jpg"),
    output_image_path=Path("output.jpg"),
    manual_bbox=(100, 100, 200, 150)  # (x1, y1, x2, y2)
)

# For multiple watermarks
watermark_remover.run_image(
    input_image_path=Path("input.jpg"),
    output_image_path=Path("output.jpg"),
    manual_bbox=[(100, 100, 200, 150), (300, 300, 400, 350)]
)
```

### Progress Callback

```python
def progress_handler(progress: int):
    print(f"Progress: {progress}%")

watermark_remover.run_image(
    input_image_path=Path("input.jpg"),
    output_image_path=Path("output.jpg"),
    progress_callback=progress_handler
)
```

## Available Models

### LAMA (Default)
- **Speed**: Fast
- **Quality**: Good
- **Best for**: General purpose watermark removal
- **Description**: Large Mask Inpainting model, provides excellent results with quick processing

### MAT
- **Speed**: Moderate
- **Quality**: Good
- **Best for**: Alternative inpainting approach
- **Description**: Mask-Aware Transformer for high-quality image inpainting

## Datasets

The labeled datasets are available on Hugging Face: [sora-watermark-dataset](https://huggingface.co/datasets/LLinked/sora-watermark-dataset). Feel free to train your custom detector model or improve our model!

## License

Apache License

## Citation

If you use this project, please cite:

```bibtex
@misc{imagewatermarkremover2025,
  author = {linkedlist771},
  title = {Image Watermark Remover},
  year = {2025},
  url = {https://github.com/linkedlist771/SoraWatermarkCleaner}
}
```

## Acknowledgments

- [IOPaint](https://github.com/Sanster/IOPaint) for the LAMA and MAT implementation
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
