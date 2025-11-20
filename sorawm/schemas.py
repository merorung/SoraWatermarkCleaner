from enum import StrEnum


class CleanerType(StrEnum):
    LAMA = "lama"
    MAT = "mat"
    E2FGVI = "e2fgvi"  # Faster version
    E2FGVI_HQ = "e2fgvi_hq"  # Highest quality
