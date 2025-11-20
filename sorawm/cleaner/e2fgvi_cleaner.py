from pathlib import Path

from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig
from sorawm.configs import E2FGVI_HQ_CHECKPOINT_PATH


class E2FGVIConfig(E2FGVIHDConfig):
    """Faster E2FGVI configuration with reduced quality and lower VRAM usage"""
    ref_length: int = 25  # Increased from 10 (less reference frames = faster, less VRAM)
    num_ref: int = 2  # Very limited reference frames (was -1 = all frames)
    neighbor_stride: int = 12  # Increased from 5 (skip more frames = faster, less VRAM)
    chunk_size_ratio: float = 0.08  # Reduced from 0.15 (much smaller chunks for high-res videos)
    overlap_ratio: float = 0.01  # Reduced from 0.02 (minimal overlap)


class E2FGVICleaner(E2FGVIHDCleaner):
    """E2FGVI - Faster version with slightly lower quality than HQ

    Good balance between speed and quality:
    - 2-3x faster than E2FGVI-HQ
    - Still maintains temporal consistency
    - Better quality than MAT
    """

    def __init__(
        self,
        ckpt_path: Path = E2FGVI_HQ_CHECKPOINT_PATH,
        config: E2FGVIConfig = E2FGVIConfig(),
    ):
        # Reuse same model as HQ but with different config
        super().__init__(ckpt_path=ckpt_path, config=config)
