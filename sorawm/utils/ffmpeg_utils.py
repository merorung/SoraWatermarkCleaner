"""
FFmpeg utilities for detecting and configuring FFmpeg paths.

This module provides functions to detect FFmpeg executables in the local
project directory or system PATH, enabling portable FFmpeg usage.
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def find_local_ffmpeg() -> tuple[Optional[str], Optional[str]]:
    """
    Find FFmpeg executables in the local project directory.

    Returns:
        tuple[Optional[str], Optional[str]]: Paths to ffmpeg and ffprobe executables,
                                              or (None, None) if not found.
    """
    project_root = get_project_root()
    ffmpeg_dir = project_root / "ffmpeg"

    # Determine executable extension based on platform
    exe_ext = ".exe" if os.name == "nt" else ""

    ffmpeg_path = ffmpeg_dir / f"ffmpeg{exe_ext}"
    ffprobe_path = ffmpeg_dir / f"ffprobe{exe_ext}"

    if ffmpeg_path.exists() and ffprobe_path.exists():
        logger.debug(f"Found local FFmpeg at: {ffmpeg_path}")
        logger.debug(f"Found local FFprobe at: {ffprobe_path}")
        return str(ffmpeg_path.absolute()), str(ffprobe_path.absolute())

    return None, None


def find_system_ffmpeg() -> tuple[Optional[str], Optional[str]]:
    """
    Find FFmpeg executables in system PATH.

    Returns:
        tuple[Optional[str], Optional[str]]: Paths to ffmpeg and ffprobe executables,
                                              or (None, None) if not found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path and ffprobe_path:
        logger.debug(f"Found system FFmpeg at: {ffmpeg_path}")
        logger.debug(f"Found system FFprobe at: {ffprobe_path}")
        return ffmpeg_path, ffprobe_path

    return None, None


def get_ffmpeg_path() -> tuple[Optional[str], Optional[str]]:
    """
    Get FFmpeg executable paths, preferring local installation over system.

    Returns:
        tuple[Optional[str], Optional[str]]: Paths to ffmpeg and ffprobe executables.

    Raises:
        RuntimeError: If FFmpeg is not found in local directory or system PATH.
    """
    # Try local FFmpeg first
    ffmpeg_path, ffprobe_path = find_local_ffmpeg()
    if ffmpeg_path and ffprobe_path:
        return ffmpeg_path, ffprobe_path

    # Fall back to system FFmpeg
    ffmpeg_path, ffprobe_path = find_system_ffmpeg()
    if ffmpeg_path and ffprobe_path:
        logger.info("Using system FFmpeg (local FFmpeg not found)")
        return ffmpeg_path, ffprobe_path

    raise RuntimeError(
        "FFmpeg not found! Please either:\n"
        "1. Place ffmpeg and ffprobe executables in the 'ffmpeg/' directory, or\n"
        "2. Install FFmpeg system-wide and add it to your PATH.\n"
        "See ffmpeg/README.md for detailed instructions."
    )


def configure_ffmpeg_environment() -> dict[str, str]:
    """
    Configure environment variables for FFmpeg.

    This function sets up the necessary environment variables to ensure
    ffmpeg-python uses the correct FFmpeg executables.

    Returns:
        dict[str, str]: Dictionary of environment variables to set.
    """
    try:
        ffmpeg_path, ffprobe_path = get_ffmpeg_path()

        # Set environment variables for ffmpeg-python
        env_vars = {
            "FFMPEG_BINARY": ffmpeg_path,
            "FFPROBE_BINARY": ffprobe_path,
        }

        # Apply to current environment
        os.environ.update(env_vars)

        logger.info(f"FFmpeg configured: {ffmpeg_path}")
        logger.info(f"FFprobe configured: {ffprobe_path}")

        return env_vars
    except RuntimeError as e:
        logger.error(str(e))
        raise


def verify_ffmpeg_installation() -> bool:
    """
    Verify that FFmpeg is properly installed and working.

    Returns:
        bool: True if FFmpeg is working correctly, False otherwise.
    """
    try:
        ffmpeg_path, ffprobe_path = get_ffmpeg_path()

        # Test ffmpeg
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            logger.error("FFmpeg executable found but not working correctly")
            return False

        # Test ffprobe
        result = subprocess.run(
            [ffprobe_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            logger.error("FFprobe executable found but not working correctly")
            return False

        logger.info("FFmpeg installation verified successfully")
        return True

    except Exception as e:
        logger.error(f"FFmpeg verification failed: {e}")
        return False


def get_ffmpeg_version() -> Optional[str]:
    """
    Get the version of installed FFmpeg.

    Returns:
        Optional[str]: FFmpeg version string, or None if not available.
    """
    try:
        ffmpeg_path, _ = get_ffmpeg_path()
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # Extract version from first line
            first_line = result.stdout.split('\n')[0]
            return first_line

        return None

    except Exception as e:
        logger.error(f"Failed to get FFmpeg version: {e}")
        return None
