from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from tqdm import tqdm

import ffmpeg
from sorawm.schemas import CleanerType
from sorawm.utils.ffmpeg_utils import configure_ffmpeg_environment, get_ffmpeg_path

# Get FFmpeg binary paths for use in ffmpeg-python calls
try:
    _FFMPEG_BIN, _FFPROBE_BIN = get_ffmpeg_path()
except Exception:
    _FFMPEG_BIN, _FFPROBE_BIN = 'ffmpeg', 'ffprobe'
from sorawm.utils.imputation_utils import (find_2d_data_bkps,
                                           find_idxs_interval,
                                           get_interval_average_bbox)
from sorawm.utils.video_utils import VideoLoader, merge_frames_with_overlap
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]


class SoraWM:
    def __init__(self, cleaner_type: CleanerType = CleanerType.LAMA):
        # Configure FFmpeg to use local installation if available
        try:
            configure_ffmpeg_environment()
        except RuntimeError as e:
            logger.error(f"FFmpeg configuration failed: {e}")
            raise

        self.detector = SoraWaterMarkDetector()
        self.cleaner = WaterMarkCleaner(cleaner_type)
        self.cleaner_type = cleaner_type

    def run_batch(
        self,
        input_video_dir_path: Path,
        output_video_dir_path: Path | None = None,
        progress_callback: Callable[[int], None] | None = None,
        quiet: bool = False,
    ):
        if output_video_dir_path is None:
            output_video_dir_path = input_video_dir_path.parent / "watermark_removed"
            if not quiet:
                logger.warning(
                    f"output_video_dir_path is not set, using {output_video_dir_path} as output_video_dir_path"
                )
        output_video_dir_path.mkdir(parents=True, exist_ok=True)
        input_video_paths = []
        for ext in VIDEO_EXTENSIONS:
            input_video_paths.extend(input_video_dir_path.rglob(f"*{ext}"))

        video_lengths = len(input_video_paths)
        if not quiet:
            logger.info(f"Found {video_lengths} video(s) to process")
        for idx, input_video_path in enumerate(
            tqdm(input_video_paths, desc="Processing videos", disable=quiet)
        ):
            output_video_path = output_video_dir_path / input_video_path.name
            if progress_callback:

                def batch_progress_callback(single_video_progress: int):
                    overall_progress = int(
                        (idx / video_lengths) * 100
                        + (single_video_progress / video_lengths)
                    )
                    progress_callback(min(overall_progress, 100))

                self.run(
                    input_video_path,
                    output_video_path,
                    progress_callback=batch_progress_callback,
                    quiet=quiet,
                )
            else:
                self.run(
                    input_video_path,
                    output_video_path,
                    progress_callback=None,
                    quiet=quiet,
                )

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
        quiet: bool = False,
        manual_bbox: tuple[int, int, int, int] | list[tuple[int, int, int, int]] | None = None,
    ):
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames

        temp_output_path = output_video_path.parent / f"temp_{output_video_path.name}"
        output_options = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "preset": "slow",
        }

        if input_video_loader.original_bitrate:
            output_options["video_bitrate"] = str(
                int(int(input_video_loader.original_bitrate) * 1.2)
            )
        else:
            output_options["crf"] = "18"

        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), **output_options)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True, cmd=_FFMPEG_BIN)
        )

        frame_bboxes = {}
        detect_missed = []
        bbox_centers = []
        bboxes = []

        if not quiet:
            logger.debug(
                f"total frames: {total_frames}, fps: {fps}, width: {width}, height: {height}"
            )

        # Use manual bbox if provided
        if manual_bbox is not None:
            # Support both single bbox and multiple bboxes
            if isinstance(manual_bbox, list):
                # Multiple bboxes: merge them into one combined mask
                if not quiet:
                    logger.info(f"Using {len(manual_bbox)} manual bboxes: {manual_bbox}")
                # For each frame, store the list of bboxes
                for idx in range(total_frames):
                    frame_bboxes[idx] = {"bbox": manual_bbox}
            else:
                # Single bbox
                if not quiet:
                    logger.info(f"Using manual bbox: {manual_bbox}")
                # Fill all frames with the same bbox
                for idx in range(total_frames):
                    frame_bboxes[idx] = {"bbox": manual_bbox}
            # Skip detection progress and jump directly to 50%
            if progress_callback:
                if progress_callback(50) == False:
                    raise InterruptedError("Processing cancelled by user")
        else:
            # Auto detection mode
            for idx, frame in enumerate(
                tqdm(
                    input_video_loader,
                    total=total_frames,
                    desc="Detect watermarks",
                    disable=quiet,
                )
            ):
                detection_result = self.detector.detect(frame)
                if detection_result["detected"]:
                    frame_bboxes[idx] = {"bbox": detection_result["bbox"]}
                    x1, y1, x2, y2 = detection_result["bbox"]
                    bbox_centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                    bboxes.append((x1, y1, x2, y2))

                else:
                    frame_bboxes[idx] = {"bbox": None}
                    detect_missed.append(idx)
                    bbox_centers.append(None)
                    bboxes.append(None)
                # 10% - 50%
                if progress_callback and idx % 10 == 0:
                    progress = 10 + int((idx / total_frames) * 40)
                    if progress_callback(progress) == False:
                        raise InterruptedError("Processing cancelled by user")
        if not quiet:
            logger.debug(f"detect missed frames: {detect_missed}")
        bkps_full = [0, total_frames]

        # Skip imputation if using manual bbox
        if manual_bbox is None and detect_missed:
            # 1. find the bkps of the bbox centers
            bkps = find_2d_data_bkps(bbox_centers)
            # add the start and end position, to form the complete interval boundaries
            bkps_full = [0] + bkps + [total_frames]
            # bkps_full = bkps_full[0] + bkps + bkps_full[1]
            # logger.debug(f"bkps intervals: {bkps_full}")

            # 2. calculate the average bbox of each interval
            interval_bboxes = get_interval_average_bbox(bboxes, bkps_full)
            # logger.debug(f"interval average bboxes: {interval_bboxes}")

            # 3. find the interval index of each missed frame
            missed_intervals = find_idxs_interval(detect_missed, bkps_full)
            # logger.debug(
            #     f"missed frame intervals: {list(zip(detect_missed, missed_intervals))}"
            # )

            # 4. fill the missed frames with the average bbox of the corresponding interval
            for missed_idx, interval_idx in zip(detect_missed, missed_intervals):
                if (
                    interval_idx < len(interval_bboxes)
                    and interval_bboxes[interval_idx] is not None
                ):
                    frame_bboxes[missed_idx]["bbox"] = interval_bboxes[interval_idx]
                    if not quiet:
                        logger.debug(
                            f"Filled missed frame {missed_idx} with bbox:\n"
                            f" {interval_bboxes[interval_idx]}"
                        )
                else:
                    # if the interval has no valid bbox, use the previous and next frame to complete (fallback strategy)
                    before = max(missed_idx - 1, 0)
                    after = min(missed_idx + 1, total_frames - 1)
                    before_box = frame_bboxes[before]["bbox"]
                    after_box = frame_bboxes[after]["bbox"]
                    if before_box:
                        frame_bboxes[missed_idx]["bbox"] = before_box
                    elif after_box:
                        frame_bboxes[missed_idx]["bbox"] = after_box
        else:
            del bboxes
            del bbox_centers
            del detect_missed

        if self.cleaner_type in [CleanerType.LAMA, CleanerType.MAT]:
            ## 1. Image-based Cleaner Strategy (LAMA/MAT) with mask dilation for better quality.
            import cv2
            input_video_loader = VideoLoader(input_video_path)
            # Kernel size for mask dilation (larger = more aggressive cleaning, but may affect surrounding area)
            # 15-20 is good balance for Sora watermarks
            kernel_size = 17
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            for idx, frame in enumerate(
                tqdm(
                    input_video_loader,
                    total=total_frames,
                    desc="Remove watermarks",
                    disable=quiet,
                )
            ):
                bbox = frame_bboxes[idx]["bbox"]
                if bbox is not None:
                    mask = np.zeros((height, width), dtype=np.uint8)

                    # Handle multiple bboxes
                    if isinstance(bbox, list):
                        # Multiple bboxes: merge into single mask
                        for x1, y1, x2, y2 in bbox:
                            mask[y1:y2, x1:x2] = 255
                    else:
                        # Single bbox
                        x1, y1, x2, y2 = bbox
                        mask[y1:y2, x1:x2] = 255

                    # Dilate mask to cover more area around watermark edges
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    cleaned_frame = self.cleaner.clean(frame, mask)
                else:
                    cleaned_frame = frame
                process_out.stdin.write(cleaned_frame.tobytes())

                # 50% - 95%
                if progress_callback and idx % 10 == 0:
                    progress = 50 + int((idx / total_frames) * 45)
                    if progress_callback(progress) == False:
                        raise InterruptedError("Processing cancelled by user")
        elif self.cleaner_type in [CleanerType.E2FGVI, CleanerType.E2FGVI_HQ]:
            ## 2. E2FGVI_HQ Cleaner Strategy with overlap blending.
            input_video_loader = VideoLoader(input_video_path)
            frame_counter = 0
            overlap_ratio = self.cleaner.config.overlap_ratio
            all_cleaned_frames = None
            # Create overlapping segments for smooth transitions
            num_segments = len(bkps_full) - 1
            for segment_idx in range(num_segments):
                seg_start = bkps_full[segment_idx]
                seg_end = bkps_full[segment_idx + 1]
                seg_length = seg_end - seg_start
                # Calculate overlap size based on segment length
                segment_overlap = max(1, int(overlap_ratio * seg_length))
                # Extend segment boundaries to create overlap (except for first/last)
                start = seg_start
                end = seg_end

                # Add overlap at the start (except for first segment)
                if segment_idx > 0:
                    start = max(seg_start - segment_overlap, bkps_full[segment_idx - 1])

                # Add overlap at the end (except for last segment)
                if segment_idx < num_segments - 1:
                    end = min(seg_end + segment_overlap, bkps_full[segment_idx + 2])

                if not quiet:
                    logger.debug(f"Segment {segment_idx}: original=[{seg_start}, {seg_end}), "
                               f"with_overlap=[{start}, {end}), overlap={segment_overlap}")

                frames = np.array(input_video_loader.get_slice(start, end))
                # Convert BGR to RGB for E2FGVI_HQ cleaner (expects RGB format)
                frames = frames[:, :, :, ::-1].copy()

                masks = np.zeros((len(frames), height, width), dtype=np.uint8)
                for idx in range(start, end):
                    bbox = frame_bboxes[idx]["bbox"]
                    if bbox is not None:
                        idx_offset = idx - start
                        # Handle multiple bboxes
                        if isinstance(bbox, list):
                            # Multiple bboxes: merge into single mask
                            for x1, y1, x2, y2 in bbox:
                                masks[idx_offset][y1:y2, x1:x2] = 255
                        else:
                            # Single bbox
                            x1, y1, x2, y2 = bbox
                            masks[idx_offset][y1:y2, x1:x2] = 255

                # Progress callback for E2FGVI_HQ processing
                # 50% - 95% range
                if progress_callback:
                    progress = 50 + int((segment_idx / num_segments) * 45)
                    if progress_callback(progress) == False:
                        raise InterruptedError("Processing cancelled by user")

                cleaned_frames = self.cleaner.clean(frames, masks)
                
                # Merge with overlap blending support
                all_cleaned_frames = merge_frames_with_overlap(
                    result_frames=all_cleaned_frames,
                    chunk_frames=cleaned_frames,
                    start_idx=start,
                    overlap_size=segment_overlap,
                    is_first_chunk=(segment_idx == 0),
                )
                
                # Determine which frames to write from this segment
                # Write the core segment (seg_start to seg_end), skip overlaps for subsequent processing
                write_start = seg_start
                write_end = seg_end
                
                for write_idx in range(write_start, write_end):
                    if write_idx < len(all_cleaned_frames) and all_cleaned_frames[write_idx] is not None:
                        cleaned_frame = all_cleaned_frames[write_idx]
                        # Convert RGB back to BGR for FFmpeg output (expects bgr24 format)
                        cleaned_frame_bgr = cleaned_frame[:, :, ::-1]
                        process_out.stdin.write(cleaned_frame_bgr.astype(np.uint8).tobytes())
                        frame_counter += 1
                        # 50% - 95%
                        if progress_callback and frame_counter % 10 == 0:
                            progress = 50 + int((frame_counter / total_frames) * 45)
                            if progress_callback(progress) == False:
                                raise InterruptedError("Processing cancelled by user")

        process_out.stdin.close()
        process_out.wait()

        # 95% - 99%
        if progress_callback:
            if progress_callback(95) == False:
                raise InterruptedError("Processing cancelled by user")

        self.merge_audio_track(input_video_path, temp_output_path, output_video_path)

        if progress_callback:
            if progress_callback(99) == False:
                raise InterruptedError("Processing cancelled by user")

    def merge_audio_track(
        self, input_video_path: Path, temp_output_path: Path, output_video_path: Path
    ):
        logger.info(f"Merging audio track to: {output_video_path}")
        logger.info(f"Temp file exists: {temp_output_path.exists()}")

        video_stream = ffmpeg.input(str(temp_output_path))
        audio_stream = ffmpeg.input(str(input_video_path)).audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_video_path),
                vcodec="copy",
                acodec="aac",
            )
            .overwrite_output()
            .run(quiet=True, cmd=_FFMPEG_BIN)
        )

        # Verify output file was created
        if output_video_path.exists():
            file_size = output_video_path.stat().st_size
            logger.info(f"✓ Successfully saved video at: {output_video_path}")
            logger.info(f"✓ File size: {file_size / (1024*1024):.2f} MB")
        else:
            logger.error(f"✗ Failed to save video at: {output_video_path}")
            raise RuntimeError(f"Output file was not created at: {output_video_path}")

        # Clean up temp file
        if temp_output_path.exists():
            temp_output_path.unlink()
            logger.info("✓ Cleaned up temporary file")

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

            # Dilate mask for better results (same as video processing)
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

    input_video_path = Path(
        "resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4"
    )
    output_video_path = Path("outputs/sora_watermark_removed.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)
