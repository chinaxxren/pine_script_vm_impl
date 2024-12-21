from typing import List, Union, Optional
from dataclasses import dataclass
from enum import Enum, auto

class TimeFrame(Enum):
    TICK = auto()
    SECOND = auto()
    MINUTE = auto()
    HOUR = auto()
    DAY = auto()
    WEEK = auto()
    MONTH = auto()

class PineType(Enum):
    """Pine Script 类型"""
    NUMBER = auto()  # 数字类型
    STRING = auto()  # 字符串类型
    BOOLEAN = auto() # 布尔类型
    COLOR = auto()   # 颜色类型
    LINE = auto()    # 线条类型
    LABEL = auto()   # 标签类型
    SERIES = auto()  # 序列类型
    NA = auto()      # NA类型

@dataclass
class PineValue:
    """Base class for Pine Script values"""
    value: Union[float, int, bool, str]
    na: bool = False

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return PineValue(self.value + other)
        if isinstance(other, PineValue):
            return PineValue(self.value + other.value)
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return PineValue(self.value - other)
        if isinstance(other, PineValue):
            return PineValue(self.value - other.value)
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return PineValue(self.value * other)
        if isinstance(other, PineValue):
            return PineValue(self.value * other.value)
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return PineValue(self.value / other)
        if isinstance(other, PineValue):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero")
            return PineValue(self.value / other.value)
        raise TypeError(f"Unsupported operand type: {type(other)}")

class Series:
    """Represents a time series in Pine Script"""
    def __init__(self, data: Optional[List[PineValue]] = None):
        self.data = data or []
        self.current_index = 0

    def append(self, value: Union[PineValue, float, int]) -> None:
        if not isinstance(value, PineValue):
            value = PineValue(value)
        self.data.append(value)

    def current(self) -> Optional[PineValue]:
        if 0 <= self.current_index < len(self.data):
            return self.data[self.current_index]
        return None

    def shift(self, offset: int = 1) -> Optional[PineValue]:
        index = self.current_index - offset
        if 0 <= index < len(self.data):
            return self.data[index]
        return PineValue(float('nan'), na=True)

    def __len__(self) -> int:
        return len(self.data)
