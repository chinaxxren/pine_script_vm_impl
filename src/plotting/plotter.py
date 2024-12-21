from enum import Enum, auto
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

class Color(Enum):
    BLACK = '#000000'
    WHITE = '#FFFFFF'
    RED = '#FF0000'
    GREEN = '#00FF00'
    BLUE = '#0000FF'
    YELLOW = '#FFFF00'
    PURPLE = '#800080'
    ORANGE = '#FFA500'
    GRAY = '#808080'

class LineStyle(Enum):
    SOLID = '-'
    DASHED = '--'
    DOTTED = ':'
    DASHDOT = '-.'

class PlotStyle(Enum):
    LINE = auto()
    HISTOGRAM = auto()
    CIRCLES = auto()
    CROSS = auto()
    AREA = auto()
    COLUMNS = auto()
    BARS = auto()
    SCATTER = auto()
    CANDLESTICK = auto()
    BAND = auto()
    MARKER = auto()
    LABEL = auto()
    ARROW = auto()
    BGCOLOR = auto()

@dataclass
class PlotOptions:
    """绘图选项"""
    title: str = ''
    color: Color = Color.BLUE
    line_style: LineStyle = LineStyle.SOLID
    style: PlotStyle = PlotStyle.LINE
    width: float = 1.0
    opacity: float = 1.0
    show_last: Optional[int] = None
    overlay: bool = False
    fill_color: Optional[str] = None
    marker_size: float = 5.0
    marker_style: str = 'circle'
    text: str = ''
    font_size: float = 12.0
    font_family: str = 'Arial'
    z_index: int = 0

class Plotter:
    def __init__(self):
        self.plots: List[Dict[str, Any]] = []
        self.figure = None
        self.axes = None

    def plot(self, data: Union[List[float], np.ndarray], options: PlotOptions) -> None:
        """添加一个新的绘图"""
        self.plots.append({
            'data': np.array(data),
            'options': options
        })

    def show(self, width: int = 12, height: int = 6) -> None:
        """显示所有绘图"""
        if not self.plots:
            return

        self.figure, self.axes = plt.subplots(figsize=(width, height))
        
        # 绘制每个图表
        for plot in self.plots:
            data = plot['data']
            options = plot['options']
            
            if options.show_last:
                data = data[-options.show_last:]
            
            x = np.arange(len(data))
            
            if options.style == PlotStyle.LINE:
                self.axes.plot(x, data, 
                             color=options.color.value,
                             linestyle=options.line_style.value,
                             linewidth=options.width,
                             alpha=options.opacity)
                
            elif options.style == PlotStyle.HISTOGRAM:
                self.axes.bar(x, data,
                            color=options.color.value,
                            alpha=options.opacity,
                            width=options.width)
                
            elif options.style == PlotStyle.CIRCLES:
                self.axes.scatter(x, data,
                                color=options.color.value,
                                alpha=options.opacity,
                                s=options.width*50)
                
            elif options.style == PlotStyle.AREA:
                self.axes.fill_between(x, data,
                                     color=options.color.value,
                                     alpha=options.opacity)
                
            elif options.style == PlotStyle.COLUMNS:
                self.axes.bar(x, data,
                            color=options.color.value,
                            alpha=options.opacity)
                
            elif options.style == PlotStyle.BARS:
                self.axes.hlines(data, x, x+1,
                               color=options.color.value,
                               linewidth=options.width,
                               alpha=options.opacity)
                
            elif options.style == PlotStyle.SCATTER:
                self.axes.scatter(x, data,
                                color=options.color.value,
                                alpha=options.opacity,
                                s=options.marker_size)
                
            elif options.style == PlotStyle.CANDLESTICK:
                self.axes.plot(x, data,
                             color=options.color.value,
                             linestyle=options.line_style.value,
                             linewidth=options.width,
                             alpha=options.opacity)
                
            elif options.style == PlotStyle.BAND:
                self.axes.fill_between(x, data[:, 0], data[:, 1],
                                     color=options.color.value,
                                     alpha=options.opacity)
                
            elif options.style == PlotStyle.MARKER:
                self.axes.scatter(x, data,
                                color=options.color.value,
                                alpha=options.opacity,
                                s=options.marker_size,
                                marker=options.marker_style)
                
            elif options.style == PlotStyle.LABEL:
                self.axes.text(x, data,
                             options.text,
                             color=options.color.value,
                             fontsize=options.font_size,
                             fontfamily=options.font_family)
                
            elif options.style == PlotStyle.ARROW:
                self.axes.arrow(x[0], data[0],
                              x[1] - x[0], data[1] - data[0],
                              color=options.color.value,
                              width=options.width,
                              head_width=options.width*2,
                              head_length=options.width*2)
                
            elif options.style == PlotStyle.BGCOLOR:
                self.axes.axvspan(x[0], x[-1],
                                facecolor=options.color.value,
                                alpha=options.opacity)

        if any(plot['options'].title for plot in self.plots):
            self.axes.set_title('\n'.join(plot['options'].title 
                                        for plot in self.plots 
                                        if plot['options'].title))
            
        self.axes.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def clear(self) -> None:
        """清除所有绘图"""
        self.plots.clear()
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.axes = None
