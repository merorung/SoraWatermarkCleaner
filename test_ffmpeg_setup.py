"""
FFmpeg Setup Test Script

This script tests whether FFmpeg is properly configured for the project.
It checks for local FFmpeg installation in the ffmpeg/ directory and falls
back to system FFmpeg if necessary.

Usage:
    python test_ffmpeg_setup.py
"""
import sys
import os
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == "win32":
    # Try to set UTF-8 mode for Windows console
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path to import sorawm modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sorawm.utils.ffmpeg_utils import (
    find_local_ffmpeg,
    find_system_ffmpeg,
    get_ffmpeg_path,
    verify_ffmpeg_installation,
    get_ffmpeg_version,
    configure_ffmpeg_environment,
)


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_result(success: bool, message: str):
    """Print a formatted result message."""
    icon = "✓" if success else "✗"
    status = "PASS" if success else "FAIL"
    print(f"[{icon}] {status}: {message}")


def test_local_ffmpeg():
    """Test for local FFmpeg installation."""
    print_header("Testing Local FFmpeg Installation")

    ffmpeg_path, ffprobe_path = find_local_ffmpeg()

    if ffmpeg_path and ffprobe_path:
        print_result(True, f"Local FFmpeg found at: {ffmpeg_path}")
        print_result(True, f"Local FFprobe found at: {ffprobe_path}")
        return True
    else:
        print_result(False, "Local FFmpeg not found in ffmpeg/ directory")
        return False


def test_system_ffmpeg():
    """Test for system FFmpeg installation."""
    print_header("Testing System FFmpeg Installation")

    ffmpeg_path, ffprobe_path = find_system_ffmpeg()

    if ffmpeg_path and ffprobe_path:
        print_result(True, f"System FFmpeg found at: {ffmpeg_path}")
        print_result(True, f"System FFprobe found at: {ffprobe_path}")
        return True
    else:
        print_result(False, "System FFmpeg not found in PATH")
        return False


def test_ffmpeg_configuration():
    """Test FFmpeg configuration."""
    print_header("Testing FFmpeg Configuration")

    try:
        configure_ffmpeg_environment()
        print_result(True, "FFmpeg environment variables configured successfully")
        return True
    except RuntimeError as e:
        print_result(False, f"FFmpeg configuration failed: {e}")
        return False


def test_ffmpeg_verification():
    """Test FFmpeg verification."""
    print_header("Testing FFmpeg Verification")

    if verify_ffmpeg_installation():
        print_result(True, "FFmpeg installation verified successfully")

        version = get_ffmpeg_version()
        if version:
            print(f"\n  Version Info: {version}")

        return True
    else:
        print_result(False, "FFmpeg verification failed")
        return False


def main():
    """Run all FFmpeg tests."""
    print("\n" + "="*60)
    print("  SoraWatermarkCleaner - FFmpeg Setup Test")
    print("="*60)

    # Track test results
    results = []

    # Test 1: Check local FFmpeg
    has_local = test_local_ffmpeg()
    results.append(("Local FFmpeg", has_local))

    # Test 2: Check system FFmpeg
    has_system = test_system_ffmpeg()
    results.append(("System FFmpeg", has_system))

    # If neither is available, exit early
    if not has_local and not has_system:
        print_header("Test Summary")
        print_result(False, "No FFmpeg installation found!")
        print("\nPlease install FFmpeg by following one of these methods:")
        print("  1. Download FFmpeg executables to ffmpeg/ directory (recommended)")
        print("     See ffmpeg/README.md for detailed instructions")
        print("  2. Install FFmpeg system-wide and add to PATH")
        return False

    # Test 3: Configuration
    config_success = test_ffmpeg_configuration()
    results.append(("FFmpeg Configuration", config_success))

    # Test 4: Verification
    verify_success = test_ffmpeg_verification()
    results.append(("FFmpeg Verification", verify_success))

    # Print summary
    print_header("Test Summary")

    all_passed = all(result for _, result in results if _ not in ["Local FFmpeg", "System FFmpeg"])
    has_ffmpeg = has_local or has_system

    if all_passed and has_ffmpeg:
        if has_local:
            print_result(True, "Using LOCAL FFmpeg (portable mode)")
        else:
            print_result(True, "Using SYSTEM FFmpeg")

        print_result(True, "All tests passed! FFmpeg is correctly configured")
        print("\n✓ You can now use the SoraWatermarkCleaner project")
        return True
    else:
        print_result(False, "Some tests failed")
        print("\nPlease check the errors above and:")
        print("  1. Ensure FFmpeg executables are properly installed")
        print("  2. Check file permissions (especially on macOS/Linux)")
        print("  3. Verify FFmpeg executables are not corrupted")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
