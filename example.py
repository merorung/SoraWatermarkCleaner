from pathlib import Path

from sorawm.core import SoraWM
from sorawm.schemas import CleanerType

if __name__ == "__main__":
    input_image_path = Path("resources/watermark_template.png")
    output_image_path = Path("outputs/watermark_removed")

    # 1. LAMA is fast and good quality
    sora_wm = SoraWM(cleaner_type=CleanerType.LAMA)
    sora_wm.run_image(input_image_path, Path(f"{output_image_path}_lama.png"))

    # 2. MAT is another option for image inpainting
    sora_wm = SoraWM(cleaner_type=CleanerType.MAT)
    sora_wm.run_image(input_image_path, Path(f"{output_image_path}_mat.png"))
