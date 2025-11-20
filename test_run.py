from pathlib import Path
from sorawm.core import SoraWM
from sorawm.schemas import CleanerType

if __name__ == "__main__":
    input_video_path = Path("resources/dog_vs_sam.mp4")
    output_video_path = Path("outputs/sora_watermark_removed_lama.mp4")

    print("Starting watermark removal with LAMA model...")
    print(f"Input: {input_video_path}")
    print(f"Output: {output_video_path}")

    # Use LAMA (fast and good quality)
    sora_wm = SoraWM(cleaner_type=CleanerType.LAMA)
    sora_wm.run(input_video_path, output_video_path)

    print(f"\nDone! Check the output at: {output_video_path}")
