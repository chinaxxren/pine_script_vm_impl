from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.lines import Line2D
import mplfinance as mpf
from datetime import datetime

class ChartType(Enum):
    """图表类型"""
    CANDLE = auto()      # K线图
    LINE = auto()        # 线图
    AREA = auto()        # 面积图
    RENKO = auto()       # 砖形图
    POINT_FIGURE = auto()  # 点数图

class ChartEvent(Enum):
    """图表事件"""
    CLICK = auto()         # 点击
    DOUBLE_CLICK = auto()  # 双击
    HOVER = auto()         # 悬停
    ZOOM = auto()         # 缩放
    PAN = auto()          # 平移
    SELECT = auto()       # 选择
    CROSSHAIR = auto()    # 十字光标

class ChartTool(Enum):
    """图表工具"""
    CURSOR = auto()       # 光标
    CROSSHAIR = auto()    # 十字光标
    LINE = auto()         # 线段
    RAY = auto()         # 射线
    ARROW = auto()       # 箭头
    RECT = auto()        # 矩形
    CIRCLE = auto()      # 圆形
    TEXT = auto()        # 文本
    FIBO = auto()        # 斐波那契
    MEASURE = auto()     # 测量

class ChartData:
    """图表数据"""
    
    def __init__(self):
        self.datetime: List[datetime] = []
        self.open: List[float] = []
        self.high: List[float] = []
        self.low: List[float] = []
        self.close: List[float] = []
        self.volume: List[float] = []
        self.indicators: Dict[str, np.ndarray] = {}
        self.drawings: List[Dict[str, Any]] = []
        
class ChartInteraction:
    """图表交互"""
    
    def __init__(self, figure: Figure, axes: Axes):
        self.figure = figure
        self.axes = axes
        self.current_tool = ChartTool.CURSOR
        self.drawings: List[Dict[str, Any]] = []
        self.selected_drawing = None
        self.start_point = None
        self.temp_artist = None
        self.crosshair_lines = None
        
        # 注册事件处理器
        self.figure.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.figure.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.figure.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.figure.canvas.mpl_connect('key_press_event', self._on_key_press)
        
    def set_tool(self, tool: ChartTool) -> None:
        """设置当前工具"""
        self.current_tool = tool
        if tool == ChartTool.CROSSHAIR:
            self._init_crosshair()
        else:
            self._remove_crosshair()
            
    def _init_crosshair(self) -> None:
        """初始化十字光标"""
        self.crosshair_lines = (
            self.axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5),
            self.axes.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        )
        self.crosshair_lines[0].set_visible(False)
        self.crosshair_lines[1].set_visible(False)
        
    def _remove_crosshair(self) -> None:
        """移除十字光标"""
        if self.crosshair_lines:
            self.crosshair_lines[0].remove()
            self.crosshair_lines[1].remove()
            self.crosshair_lines = None
            
    def _on_mouse_press(self, event) -> None:
        """鼠标按下事件处理"""
        if event.inaxes != self.axes:
            return
            
        if event.button == 1:  # 左键
            if self.current_tool == ChartTool.CURSOR:
                # 选择绘图对象
                self._select_drawing(event)
            else:
                # 开始绘制
                self.start_point = (event.xdata, event.ydata)
                
    def _on_mouse_release(self, event) -> None:
        """鼠标释放事件处理"""
        if event.inaxes != self.axes or not self.start_point:
            return
            
        end_point = (event.xdata, event.ydata)
        
        if self.current_tool == ChartTool.LINE:
            self._add_line(self.start_point, end_point)
        elif self.current_tool == ChartTool.RAY:
            self._add_ray(self.start_point, end_point)
        elif self.current_tool == ChartTool.ARROW:
            self._add_arrow(self.start_point, end_point)
        elif self.current_tool == ChartTool.RECT:
            self._add_rectangle(self.start_point, end_point)
        elif self.current_tool == ChartTool.CIRCLE:
            self._add_circle(self.start_point, end_point)
        elif self.current_tool == ChartTool.FIBO:
            self._add_fibonacci(self.start_point, end_point)
            
        self.start_point = None
        if self.temp_artist:
            self.temp_artist.remove()
            self.temp_artist = None
            
        self.figure.canvas.draw()
        
    def _on_mouse_move(self, event) -> None:
        """鼠标移动事件处理"""
        if event.inaxes != self.axes:
            return
            
        if self.current_tool == ChartTool.CROSSHAIR:
            self._update_crosshair(event)
        elif self.start_point:
            self._update_temp_drawing(event)
            
    def _on_scroll(self, event) -> None:
        """滚轮事件处理"""
        if event.inaxes != self.axes:
            return
            
        # 缩放
        factor = 0.9 if event.button == 'up' else 1.1
        self.axes.set_xlim([x * factor for x in self.axes.get_xlim()])
        self.axes.set_ylim([y * factor for y in self.axes.get_ylim()])
        self.figure.canvas.draw()
        
    def _on_key_press(self, event) -> None:
        """键盘事件处理"""
        if event.key == 'delete' and self.selected_drawing:
            self._remove_selected_drawing()
            
    def _select_drawing(self, event) -> None:
        """选择绘图对象"""
        min_dist = float('inf')
        selected = None
        
        for drawing in self.drawings:
            dist = self._distance_to_drawing(drawing, event)
            if dist < min_dist:
                min_dist = dist
                selected = drawing
                
        if min_dist < 5:  # 像素阈值
            self.selected_drawing = selected
            self._highlight_selected()
        else:
            self.selected_drawing = None
            self._unhighlight_all()
            
    def _distance_to_drawing(self, drawing: Dict[str, Any], event) -> float:
        """计算点到绘图对象的距离"""
        if drawing['type'] == 'line':
            return self._distance_to_line(
                drawing['start'], drawing['end'],
                (event.xdata, event.ydata)
            )
        elif drawing['type'] == 'circle':
            return self._distance_to_circle(
                drawing['center'], drawing['radius'],
                (event.xdata, event.ydata)
            )
        return float('inf')
        
    def _distance_to_line(self, start: Tuple[float, float],
                         end: Tuple[float, float],
                         point: Tuple[float, float]) -> float:
        """计算点到线段的距离"""
        x0, y0 = point
        x1, y1 = start
        x2, y2 = end
        
        # 线段向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 如果线段长度为0
        if dx == 0 and dy == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            
        # 参数t表示投影点在线段上的位置
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)
        
        if t < 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        elif t > 1:
            return np.sqrt((x0 - x2)**2 + (y0 - y2)**2)
            
        # 投影点坐标
        px = x1 + t * dx
        py = y1 + t * dy
        
        return np.sqrt((x0 - px)**2 + (y0 - py)**2)
        
    def _add_line(self, start: Tuple[float, float],
                  end: Tuple[float, float]) -> None:
        """添加线段"""
        line = self.axes.plot(
            [start[0], end[0]], [start[1], end[1]],
            'b-', picker=5
        )[0]
        
        self.drawings.append({
            'type': 'line',
            'start': start,
            'end': end,
            'artist': line
        })
        
    def _add_fibonacci(self, start: Tuple[float, float],
                      end: Tuple[float, float]) -> None:
        """添加斐波那契回调线"""
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        
        y1, y2 = start[1], end[1]
        height = y2 - y1
        
        for level, color in zip(levels, colors):
            y = y1 + height * level
            line = self.axes.axhline(
                y=y, color=color, linestyle='--', alpha=0.5
            )
            text = self.axes.text(
                start[0], y, f'{level*100:.1f}%',
                verticalalignment='bottom'
            )
            
            self.drawings.append({
                'type': 'fibonacci',
                'level': level,
                'line': line,
                'text': text
            })
            
    def _update_crosshair(self, event) -> None:
        """更新十字光标"""
        if self.crosshair_lines:
            self.crosshair_lines[0].set_ydata(event.ydata)
            self.crosshair_lines[1].set_xdata(event.xdata)
            self.crosshair_lines[0].set_visible(True)
            self.crosshair_lines[1].set_visible(True)
            self.figure.canvas.draw()
            
    def _update_temp_drawing(self, event) -> None:
        """更新临时绘图"""
        if not self.start_point:
            return
            
        end_point = (event.xdata, event.ydata)
        
        if self.temp_artist:
            self.temp_artist.remove()
            
        if self.current_tool == ChartTool.LINE:
            self.temp_artist = self.axes.plot(
                [self.start_point[0], end_point[0]],
                [self.start_point[1], end_point[1]],
                'b--'
            )[0]
        elif self.current_tool == ChartTool.RECT:
            width = end_point[0] - self.start_point[0]
            height = end_point[1] - self.start_point[1]
            self.temp_artist = Rectangle(
                self.start_point, width, height,
                fill=False, linestyle='--'
            )
            self.axes.add_patch(self.temp_artist)
            
        self.figure.canvas.draw()
        
    def _highlight_selected(self) -> None:
        """高亮选中的绘图对象"""
        if self.selected_drawing:
            artist = self.selected_drawing['artist']
            artist.set_color('red')
            artist.set_linewidth(2)
            self.figure.canvas.draw()
            
    def _unhighlight_all(self) -> None:
        """取消所有高亮"""
        for drawing in self.drawings:
            artist = drawing['artist']
            artist.set_color('blue')
            artist.set_linewidth(1)
        self.figure.canvas.draw()
        
    def _remove_selected_drawing(self) -> None:
        """移除选中的绘图对象"""
        if self.selected_drawing:
            self.selected_drawing['artist'].remove()
            self.drawings.remove(self.selected_drawing)
            self.selected_drawing = None
            self.figure.canvas.draw()

class Chart:
    """图表"""
    
    def __init__(self, chart_type: ChartType = ChartType.CANDLE):
        self.type = chart_type
        self.data = ChartData()
        
        # 创建图表
        self.figure, self.axes = plt.subplots(figsize=(12, 8))
        self.interaction = ChartInteraction(self.figure, self.axes)
        
        # 设置样式
        self.axes.grid(True)
        self.axes.set_title("Price Chart")
        
    def update(self) -> None:
        """更新图表"""
        self.axes.clear()
        
        if self.type == ChartType.CANDLE:
            self._plot_candlestick()
        elif self.type == ChartType.LINE:
            self._plot_line()
        elif self.type == ChartType.AREA:
            self._plot_area()
            
        # 绘制指标
        for name, values in self.data.indicators.items():
            self.axes.plot(values, label=name)
            
        # 添加图例
        self.axes.legend()
        
        # 刷新画布
        self.figure.canvas.draw()
        
    def _plot_candlestick(self) -> None:
        """绘制K线图"""
        data = {
            'Date': self.data.datetime,
            'Open': self.data.open,
            'High': self.data.high,
            'Low': self.data.low,
            'Close': self.data.close,
            'Volume': self.data.volume
        }
        mpf.plot(data, type='candle', ax=self.axes)
