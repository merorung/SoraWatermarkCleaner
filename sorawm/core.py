from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from sorawm.schemas import CleanerType
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]


class SoraWM:
    def __init__(self, cleaner_type: CleanerType = CleanerType.LAMA):
        self.detector = SoraWaterMarkDetector()
        self.cleaner = WaterMarkCleaner(cleaner_type)
        self.cleaner_type = cleaner_type

    def run_image(
        self,
        input_image_path: Path,
        output_image_path: Path,
        progress_callback: Callable[[int], None] | None = None,
        quiet: bool = False,
        manual_bbox: tuple[int, int, int, int] | list[tuple[int, int, int, int]] | None = None,
    ):
        """Process a single image to remove watermarks

        Args:
            input_image_path: Path to input image
            output_image_path: Path to save processed image
            progress_callback: Optional callback for progress updates (0-100)
            quiet: If True, suppress log output
            manual_bbox: Optional manual bounding box(es) for watermark location
        """
        import cv2

        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        # Read image
        if progress_callback:
            progress_callback(10)

        image = cv2.imread(str(input_image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {input_image_path}")

        height, width = image.shape[:2]

        if not quiet:
            logger.debug(f"Image size: width={width}, height={height}")

        # Detect or use manual bbox
        if manual_bbox is not None:
            if isinstance(manual_bbox, list):
                bbox = manual_bbox
                if not quiet:
                    logger.info(f"Using {len(manual_bbox)} manual bboxes: {manual_bbox}")
            else:
                bbox = manual_bbox
                if not quiet:
                    logger.info(f"Using manual bbox: {manual_bbox}")
            if progress_callback:
                progress_callback(30)
        else:
            # Auto detection
            if progress_callback:
                progress_callback(20)

            detection_result = self.detector.detect(image)
            if detection_result["detected"]:
                bbox = detection_result["bbox"]
                if not quiet:
                    logger.info(f"Detected watermark at: {bbox}")
            else:
                if not quiet:
                    logger.warning("No watermark detected in image")
                bbox = None

            if progress_callback:
                progress_callback(40)

        # Clean image
        if bbox is not None:
            if progress_callback:
                progress_callback(50)

            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # Handle multiple bboxes
            if isinstance(bbox, list):
                for x1, y1, x2, y2 in bbox:
                    mask[y1:y2, x1:x2] = 255
            else:
                x1, y1, x2, y2 = bbox
                mask[y1:y2, x1:x2] = 255

            # Dilate mask for better results
            if self.cleaner_type in [CleanerType.LAMA, CleanerType.MAT]:
                kernel_size = 17
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask = cv2.dilate(mask, kernel, iterations=1)

            if progress_callback:
                progress_callback(60)

            # Clean with selected model
            if not quiet:
                logger.info(f"Cleaning image with {self.cleaner_type} model...")

            cleaned_image = self.cleaner.clean(image, mask)

            if progress_callback:
                progress_callback(90)
        else:
            # No watermark detected, return original
            cleaned_image = image
            if progress_callback:
                progress_callback(90)

        # Save image
        cv2.imwrite(str(output_image_path), cleaned_image)

        if not quiet:
            file_size = output_image_path.stat().st_size
            logger.info(f"✓ Successfully saved image at: {output_image_path}")
            logger.info(f"✓ File size: {file_size / 1024:.2f} KB")

        if progress_callback:
            progress_callback(100)


if __name__ == "__main__":
    from pathlib import Path

    input_image_path = Path("resources/watermark_template.png")
    output_image_path = Path("outputs/watermark_removed.png")
    sora_wm = SoraWM()
    sora_wm.run_image(input_image_path, output_image_path)
