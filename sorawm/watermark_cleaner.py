from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np

from sorawm.cleaner.lama_cleaner import LamaCleaner
from sorawm.cleaner.mat_cleaner import MATCleaner
from sorawm.schemas import CleanerType


class WaterMarkCleaner:
    """工厂类：根据 cleaner_type 返回对应的 cleaner 실例"""

    def __new__(cls, cleaner_type: CleanerType):
        """使用 __new__ 方法实现简单工厂模式"""
        match cleaner_type:
            case CleanerType.LAMA:
                return LamaCleaner()
            case CleanerType.MAT:
                return MATCleaner()
            case _:
                raise ValueError(f"Invalid cleaner type: {cleaner_type}")
