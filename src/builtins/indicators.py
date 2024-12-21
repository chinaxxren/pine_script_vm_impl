from typing import Dict, Callable, Any, List, Optional
from ..types.pine_types import Series, PineValue

# 存储技术指标的字典
_indicators: Dict[str, Callable] = {}

def register_indicator(name: str, func: Callable) -> None:
    """注册一个技术指标"""
    _indicators[name] = func

def get_indicator(name: str) -> Optional[Callable]:
    """获取技术指标函数"""
    return _indicators.get(name)

def rsi(series: Series, length: int = 14) -> Series:
    """相对强弱指标(RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = 平均上涨点数 / 平均下跌点数
    """
    output = Series()
    if len(series.data) < length + 1:
        return output

    for i in range(len(series.data)):
        if i < length:
            output.append(PineValue(float('nan'), na=True))
            continue

        gains = []
        losses = []
        for j in range(i - length + 1, i + 1):
            change = series.data[j].value - series.data[j-1].value
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / length
        avg_loss = sum(losses) / length

        if avg_loss == 0:
            rsi_value = 100
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100 - (100 / (1 + rs))

        output.append(PineValue(rsi_value))

    return output

def macd(series: Series, fast_length: int = 12, slow_length: int = 26, signal_length: int = 9) -> tuple[Series, Series, Series]:
    """移动平均趋同散度(MACD)
    返回: (macd线, signal线, histogram)
    """
    def ema(data: List[float], length: int) -> List[float]:
        multiplier = 2 / (length + 1)
        ema_values = []
        for i, value in enumerate(data):
            if i == 0:
                ema_values.append(value)
            else:
                ema_value = (value - ema_values[-1]) * multiplier + ema_values[-1]
                ema_values.append(ema_value)
        return ema_values

    values = [x.value for x in series.data]
    
    # 计算快速和慢速EMA
    fast_ema = ema(values, fast_length)
    slow_ema = ema(values, slow_length)
    
    # 计算MACD线
    macd_line = Series()
    for i in range(len(values)):
        if i < slow_length - 1:
            macd_line.append(PineValue(float('nan'), na=True))
        else:
            macd_value = fast_ema[i] - slow_ema[i]
            macd_line.append(PineValue(macd_value))
    
    # 计算信号线
    macd_values = [x.value for x in macd_line.data if not x.na]
    signal_line_values = ema(macd_values, signal_length)
    
    signal_line = Series()
    histogram = Series()
    
    signal_index = 0
    for i in range(len(macd_line.data)):
        if macd_line.data[i].na:
            signal_line.append(PineValue(float('nan'), na=True))
            histogram.append(PineValue(float('nan'), na=True))
        else:
            signal_value = signal_line_values[signal_index]
            signal_line.append(PineValue(signal_value))
            histogram.append(PineValue(macd_line.data[i].value - signal_value))
            signal_index += 1
    
    return macd_line, signal_line, histogram

def bollinger_bands(series: Series, length: int = 20, mult: float = 2.0) -> tuple[Series, Series, Series]:
    """布林带指标
    返回: (上轨, 中轨, 下轨)
    """
    def standard_deviation(values: List[float], mean: float) -> float:
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return (squared_diff_sum / len(values)) ** 0.5

    upper_band = Series()
    middle_band = Series()
    lower_band = Series()

    for i in range(len(series.data)):
        if i < length - 1:
            upper_band.append(PineValue(float('nan'), na=True))
            middle_band.append(PineValue(float('nan'), na=True))
            lower_band.append(PineValue(float('nan'), na=True))
            continue

        window = [x.value for x in series.data[i-length+1:i+1]]
        sma = sum(window) / length
        std_dev = standard_deviation(window, sma)

        middle_band.append(PineValue(sma))
        upper_band.append(PineValue(sma + mult * std_dev))
        lower_band.append(PineValue(sma - mult * std_dev))

    return upper_band, middle_band, lower_band

# 注册所有技术指标
register_indicator('rsi', rsi)
register_indicator('macd', macd)
register_indicator('bollinger_bands', bollinger_bands)
