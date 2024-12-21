from typing import Dict, Callable, Any, List
from ..types.pine_types import Series, PineValue

# 存储内置函数的字典
_builtin_functions: Dict[str, Callable] = {}

def register_builtin_function(name: str, func: Callable) -> None:
    """注册一个内置函数"""
    _builtin_functions[name] = func

def get_builtin_function(name: str) -> Callable:
    """获取内置函数"""
    return _builtin_functions.get(name)

# 基本数学函数
def abs_function(value: PineValue) -> PineValue:
    """返回绝对值"""
    return PineValue(abs(value.value))

def max_function(a: PineValue, b: PineValue) -> PineValue:
    """返回两个值中的较大值"""
    return PineValue(max(a.value, b.value))

def min_function(a: PineValue, b: PineValue) -> PineValue:
    """返回两个值中的较小值"""
    return PineValue(min(a.value, b.value))

def round_function(value: PineValue, decimals: PineValue = PineValue(0)) -> PineValue:
    """四舍五入到指定小数位"""
    return PineValue(round(value.value, int(decimals.value)))

# 时间序列函数
def highest(series: Series, length: int) -> PineValue:
    """返回过去length个周期内的最高值"""
    if len(series.data) == 0:
        return PineValue(float('nan'), na=True)
    
    start = max(0, series.current_index - length + 1)
    values = [x.value for x in series.data[start:series.current_index + 1]]
    return PineValue(max(values)) if values else PineValue(float('nan'), na=True)

def lowest(series: Series, length: int) -> PineValue:
    """返回过去length个周期内的最低值"""
    if len(series.data) == 0:
        return PineValue(float('nan'), na=True)
    
    start = max(0, series.current_index - length + 1)
    values = [x.value for x in series.data[start:series.current_index + 1]]
    return PineValue(min(values)) if values else PineValue(float('nan'), na=True)

def sma(series: Series, length: int) -> PineValue:
    """简单移动平均"""
    if len(series.data) == 0:
        return PineValue(float('nan'), na=True)
    
    start = max(0, series.current_index - length + 1)
    values = [x.value for x in series.data[start:series.current_index + 1]]
    return PineValue(sum(values) / len(values)) if values else PineValue(float('nan'), na=True)

# 注册所有内置函数
register_builtin_function('abs', abs_function)
register_builtin_function('max', max_function)
register_builtin_function('min', min_function)
register_builtin_function('round', round_function)
register_builtin_function('highest', highest)
register_builtin_function('lowest', lowest)
register_builtin_function('sma', sma)
