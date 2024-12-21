from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
from matplotlib.lines import Line2D
from matplotlib.text import Text
from dataclasses import dataclass
from enum import Enum, auto
from scipy.interpolate import UnivariateSpline
import json

class ToolType(Enum):
    """工具类型"""
    TRENDLINE = auto()      # 趋势线
    CHANNEL = auto()        # 通道
    PITCHFORK = auto()      # 安德鲁斯叉
    GANN_FAN = auto()       # 江恩扇形
    FIBONACCI_FAN = auto()  # 斐波那契扇形
    FIBONACCI_ARC = auto()  # 斐波那契弧线
    REGRESSION = auto()     # 回归通道
    SPEED_LINES = auto()    # 速度线
    TIME_CYCLES = auto()    # 时间周期
    ELLIOTT_WAVE = auto()   # 艾略特波浪
    WAVE_LINE = auto()      # 波浪线
    HARMONIC_PATTERN = auto()  # 调和形态
    FIBONACCI_RETRACEMENT = auto()  # 斐波那契回撤
    GANN_SQUARE = auto()    # 甘氏方格
    PITCHFORK_EXTENSION = auto()  # 叉子延伸

@dataclass
class Point:
    """点"""
    x: float
    y: float
    
class DrawingTool:
    """绘图工具基类"""
    
    def __init__(self):
        self.selected = False
        self.color = 'blue'
        self.line_style = '--'
        self.alpha = 0.6
        
    def draw(self, ax):
        """绘制"""
        pass
        
    def contains(self, point: Point) -> bool:
        """判断是否包含某点"""
        pass
        
    def move(self, dx: float, dy: float):
        """移动"""
        pass
        
    def highlight(self):
        """高亮"""
        self.color = 'red'
        self.line_style = '-'
        self.alpha = 1.0
        
    def unhighlight(self):
        """取消高亮"""
        self.color = 'blue'
        self.line_style = '--'
        self.alpha = 0.6

class TrendLine(DrawingTool):
    """趋势线"""
    
    def __init__(self, start: Point, end: Point):
        super().__init__()
        self.start = start
        self.end = end
        self.extended = False  # 是否延伸
        
    def draw(self, ax):
        if self.extended:
            # 计算延伸点
            dx = self.end.x - self.start.x
            dy = self.end.y - self.start.y
            k = dy / dx if dx != 0 else float('inf')
            
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            if dx != 0:
                y1 = k * (x_min - self.start.x) + self.start.y
                y2 = k * (x_max - self.start.x) + self.start.y
                ax.plot([x_min, x_max], [y1, y2],
                       color=self.color,
                       linestyle=self.line_style,
                       alpha=self.alpha)
            else:
                ax.axvline(x=self.start.x,
                          color=self.color,
                          linestyle=self.line_style,
                          alpha=self.alpha)
        else:
            ax.plot([self.start.x, self.end.x],
                   [self.start.y, self.end.y],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
                   
    def contains(self, point: Point) -> bool:
        # 计算点到线段的距离
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        
        if dx == 0 and dy == 0:
            return False
            
        t = ((point.x - self.start.x) * dx +
             (point.y - self.start.y) * dy) / (dx * dx + dy * dy)
             
        if not self.extended and (t < 0 or t > 1):
            return False
            
        px = self.start.x + t * dx
        py = self.start.y + t * dy
        
        dist = np.sqrt((point.x - px)**2 + (point.y - py)**2)
        return dist < 5  # 像素阈值
        
    def move(self, dx: float, dy: float):
        self.start.x += dx
        self.start.y += dy
        self.end.x += dx
        self.end.y += dy

class Channel(DrawingTool):
    """平行通道"""
    
    def __init__(self, start: Point, end: Point, width: float):
        super().__init__()
        self.start = start
        self.end = end
        self.width = width
        
    def draw(self, ax):
        # 计算平行线的偏移
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
            
        # 单位法向量
        nx = -dy / length
        ny = dx / length
        
        # 上边界
        upper_start = Point(
            self.start.x + nx * self.width/2,
            self.start.y + ny * self.width/2
        )
        upper_end = Point(
            self.end.x + nx * self.width/2,
            self.end.y + ny * self.width/2
        )
        
        # 下边界
        lower_start = Point(
            self.start.x - nx * self.width/2,
            self.start.y - ny * self.width/2
        )
        lower_end = Point(
            self.end.x - nx * self.width/2,
            self.end.y - ny * self.width/2
        )
        
        # 绘制通道
        ax.plot([upper_start.x, upper_end.x],
               [upper_start.y, upper_end.y],
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        ax.plot([lower_start.x, lower_end.x],
               [lower_start.y, lower_end.y],
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        # 填充通道区域
        ax.fill_between([self.start.x, self.end.x],
                       [upper_start.y, upper_end.y],
                       [lower_start.y, lower_end.y],
                       color=self.color,
                       alpha=0.1)
                       
    def contains(self, point: Point) -> bool:
        # 检查点是否在通道内
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return False
            
        # 计算点到中心线的垂直距离
        t = ((point.x - self.start.x) * dx +
             (point.y - self.start.y) * dy) / (dx * dx + dy * dy)
             
        if t < 0 or t > 1:
            return False
            
        px = self.start.x + t * dx
        py = self.start.y + t * dy
        
        dist = np.sqrt((point.x - px)**2 + (point.y - py)**2)
        return dist <= self.width/2
        
    def move(self, dx: float, dy: float):
        self.start.x += dx
        self.start.y += dy
        self.end.x += dx
        self.end.y += dy

class Pitchfork(DrawingTool):
    """安德鲁斯叉"""
    
    def __init__(self, pivot: Point, left: Point, right: Point):
        super().__init__()
        self.pivot = pivot
        self.left = left
        self.right = right
        
    def draw(self, ax):
        # 计算中点
        midpoint = Point(
            (self.left.x + self.right.x) / 2,
            (self.left.y + self.right.y) / 2
        )
        
        # 绘制手柄
        ax.plot([self.pivot.x, midpoint.x],
               [self.pivot.y, midpoint.y],
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        # 绘制分叉
        ax.plot([self.pivot.x, self.left.x],
               [self.pivot.y, self.left.y],
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        ax.plot([self.pivot.x, self.right.x],
               [self.pivot.y, self.right.y],
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        # 计算并绘制中线的延伸
        dx = midpoint.x - self.pivot.x
        dy = midpoint.y - self.pivot.y
        
        if dx != 0:
            k = dy / dx
            x_max = ax.get_xlim()[1]
            y_ext = k * (x_max - self.pivot.x) + self.pivot.y
            
            ax.plot([self.pivot.x, x_max],
                   [self.pivot.y, y_ext],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
                   
    def contains(self, point: Point) -> bool:
        # 检查点是否在任意线段附近
        midpoint = Point(
            (self.left.x + self.right.x) / 2,
            (self.left.y + self.right.y) / 2
        )
        
        lines = [
            (self.pivot, midpoint),
            (self.pivot, self.left),
            (self.pivot, self.right)
        ]
        
        for start, end in lines:
            dx = end.x - start.x
            dy = end.y - start.y
            
            if dx == 0 and dy == 0:
                continue
                
            t = ((point.x - start.x) * dx +
                 (point.y - start.y) * dy) / (dx * dx + dy * dy)
                 
            if t < 0 or t > 1:
                continue
                
            px = start.x + t * dx
            py = start.y + t * dy
            
            dist = np.sqrt((point.x - px)**2 + (point.y - py)**2)
            if dist < 5:
                return True
                
        return False
        
    def move(self, dx: float, dy: float):
        self.pivot.x += dx
        self.pivot.y += dy
        self.left.x += dx
        self.left.y += dy
        self.right.x += dx
        self.right.y += dy

class RegressionChannel(DrawingTool):
    """回归通道"""
    
    def __init__(self, points: List[Point], std_multiplier: float = 2.0):
        super().__init__()
        self.points = points
        self.std_multiplier = std_multiplier
        self._calculate_regression()
        
    def _calculate_regression(self):
        """计算回归线和通道"""
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        
        # 线性回归
        A = np.vstack([x, np.ones_like(x)]).T
        self.slope, self.intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # 计算标准差
        y_pred = self.slope * x + self.intercept
        residuals = y - y_pred
        self.std = np.std(residuals)
        
    def draw(self, ax):
        x_min, x_max = ax.get_xlim()
        x = np.array([x_min, x_max])
        
        # 中心线
        y_center = self.slope * x + self.intercept
        ax.plot(x, y_center,
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        # 上下通道
        y_upper = y_center + self.std_multiplier * self.std
        y_lower = y_center - self.std_multiplier * self.std
        
        ax.plot(x, y_upper,
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        ax.plot(x, y_lower,
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        # 填充通道区域
        ax.fill_between(x, y_upper, y_lower,
                       color=self.color,
                       alpha=0.1)
                       
    def contains(self, point: Point) -> bool:
        # 检查点是否在通道内
        y_pred = self.slope * point.x + self.intercept
        dist = abs(point.y - y_pred)
        return dist <= self.std_multiplier * self.std
        
    def move(self, dx: float, dy: float):
        for point in self.points:
            point.x += dx
            point.y += dy
        self._calculate_regression()

class GannFan(DrawingTool):
    """江恩扇形"""
    
    def __init__(self, pivot: Point):
        super().__init__()
        self.pivot = pivot
        self.angles = [
            82.5,  # 1:8
            75.0,  # 1:4
            71.25, # 1:3
            63.75, # 1:2
            45.0,  # 1:1
            26.25, # 2:1
            18.75, # 3:1
            15.0,  # 4:1
            7.5    # 8:1
        ]
        
    def draw(self, ax):
        x_max = ax.get_xlim()[1]
        length = x_max - self.pivot.x
        
        for angle in self.angles:
            # 计算终点
            rad = np.radians(angle)
            dx = length * np.cos(rad)
            dy = length * np.sin(rad)
            
            ax.plot([self.pivot.x, self.pivot.x + dx],
                   [self.pivot.y, self.pivot.y + dy],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
                   
    def contains(self, point: Point) -> bool:
        # 检查点是否在任意线上
        x_max = point.x
        length = x_max - self.pivot.x
        
        for angle in self.angles:
            rad = np.radians(angle)
            dx = length * np.cos(rad)
            dy = length * np.sin(rad)
            
            end = Point(self.pivot.x + dx, self.pivot.y + dy)
            
            # 计算点到线的距离
            line_dx = end.x - self.pivot.x
            line_dy = end.y - self.pivot.y
            
            if line_dx == 0 and line_dy == 0:
                continue
                
            t = ((point.x - self.pivot.x) * line_dx +
                 (point.y - self.pivot.y) * line_dy) / (line_dx * line_dx + line_dy * line_dy)
                 
            if t < 0 or t > 1:
                continue
                
            px = self.pivot.x + t * line_dx
            py = self.pivot.y + t * line_dy
            
            dist = np.sqrt((point.x - px)**2 + (point.y - py)**2)
            if dist < 5:
                return True
                
        return False
        
    def move(self, dx: float, dy: float):
        self.pivot.x += dx
        self.pivot.y += dy

class WaveLine(DrawingTool):
    """波浪线"""
    
    def __init__(self, points: List[Point], smoothness: float = 0.2):
        super().__init__()
        self.points = points
        self.smoothness = smoothness
        
    def draw(self, ax):
        if len(self.points) < 2:
            return
            
        # 获取点的坐标
        x = np.array([p.x for p in self.points])
        y = np.array([p.y for p in self.points])
        
        # 使用样条插值生成平滑曲线
        t = np.linspace(0, 1, len(x))
        t_smooth = np.linspace(0, 1, len(x) * 10)
        
        spl_x = UnivariateSpline(t, x, k=3, s=self.smoothness)
        spl_y = UnivariateSpline(t, y, k=3, s=self.smoothness)
        
        x_smooth = spl_x(t_smooth)
        y_smooth = spl_y(t_smooth)
        
        ax.plot(x_smooth, y_smooth,
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
    def contains(self, point: Point) -> bool:
        # 检查点是否在曲线附近
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            # 计算点到线段的距离
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            
            if dx == 0 and dy == 0:
                continue
                
            t = ((point.x - p1.x) * dx +
                 (point.y - p1.y) * dy) / (dx * dx + dy * dy)
                 
            if t < 0 or t > 1:
                continue
                
            px = p1.x + t * dx
            py = p1.y + t * dy
            
            dist = np.sqrt((point.x - px)**2 + (point.y - py)**2)
            if dist < 5:
                return True
                
        return False
        
    def move(self, dx: float, dy: float):
        for point in self.points:
            point.x += dx
            point.y += dy

class TimeCycle(DrawingTool):
    """时间周期"""
    
    def __init__(self, center: Point, radius: float, num_cycles: int = 8):
        super().__init__()
        self.center = center
        self.radius = radius
        self.num_cycles = num_cycles
        
    def draw(self, ax):
        # 绘制圆
        circle = Circle((self.center.x, self.center.y),
                       self.radius,
                       fill=False,
                       color=self.color,
                       linestyle=self.line_style,
                       alpha=self.alpha)
        ax.add_patch(circle)
        
        # 绘制分割线
        angles = np.linspace(0, 2*np.pi, self.num_cycles+1)[:-1]
        for angle in angles:
            dx = self.radius * np.cos(angle)
            dy = self.radius * np.sin(angle)
            ax.plot([self.center.x, self.center.x + dx],
                   [self.center.y, self.center.y + dy],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
                   
    def contains(self, point: Point) -> bool:
        # 检查点是否在圆上或分割线上
        dist_to_center = np.sqrt((point.x - self.center.x)**2 +
                               (point.y - self.center.y)**2)
                               
        # 检查是否在圆上
        if abs(dist_to_center - self.radius) < 5:
            return True
            
        # 检查是否在分割线上
        if dist_to_center > self.radius:
            return False
            
        angles = np.linspace(0, 2*np.pi, self.num_cycles+1)[:-1]
        for angle in angles:
            dx = self.radius * np.cos(angle)
            dy = self.radius * np.sin(angle)
            
            # 计算点到线的距离
            line_dx = dx
            line_dy = dy
            
            t = ((point.x - self.center.x) * line_dx +
                 (point.y - self.center.y) * line_dy) / (line_dx * line_dx + line_dy * line_dy)
                 
            if t < 0 or t > 1:
                continue
                
            px = self.center.x + t * line_dx
            py = self.center.y + t * line_dy
            
            dist = np.sqrt((point.x - px)**2 + (point.y - py)**2)
            if dist < 5:
                return True
                
        return False
        
    def move(self, dx: float, dy: float):
        self.center.x += dx
        self.center.y += dy

class ElliottWave(DrawingTool):
    """艾略特波浪"""
    
    def __init__(self, points: List[Point], wave_labels: List[str]):
        super().__init__()
        self.points = points
        self.wave_labels = wave_labels
        
    def draw(self, ax):
        if len(self.points) < 2:
            return
            
        # 绘制波浪线
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        
        ax.plot(x, y,
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        # 添加波浪标签
        for point, label in zip(self.points, self.wave_labels):
            ax.text(point.x, point.y, label,
                   color=self.color,
                   alpha=self.alpha)
                   
    def contains(self, point: Point) -> bool:
        # 检查点是否在线段上
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            
            if dx == 0 and dy == 0:
                continue
                
            t = ((point.x - p1.x) * dx +
                 (point.y - p1.y) * dy) / (dx * dx + dy * dy)
                 
            if t < 0 or t > 1:
                continue
                
            px = p1.x + t * dx
            py = p1.y + t * dy
            
            dist = np.sqrt((point.x - px)**2 + (point.y - py)**2)
            if dist < 5:
                return True
                
        return False
        
    def move(self, dx: float, dy: float):
        for point in self.points:
            point.x += dx
            point.y += dy

class HarmonicPattern(DrawingTool):
    """调和形态"""
    
    def __init__(self, points: List[Point], pattern_type: str):
        super().__init__()
        self.points = points
        self.pattern_type = pattern_type
        self.ratios = self._get_ratios()
        
    def _get_ratios(self) -> Dict[str, float]:
        """获取不同形态的比率"""
        patterns = {
            'butterfly': {
                'XA': 1.0,
                'AB': 0.786,
                'BC': 0.886,
                'CD': 1.618
            },
            'bat': {
                'XA': 1.0,
                'AB': 0.382,
                'BC': 0.886,
                'CD': 2.618
            },
            'gartley': {
                'XA': 1.0,
                'AB': 0.618,
                'BC': 0.386,
                'CD': 1.272
            }
        }
        return patterns.get(self.pattern_type.lower(), {})
        
    def draw(self, ax):
        if len(self.points) < 2:
            return
            
        # 绘制线段
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        
        ax.plot(x, y,
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        # 添加标签
        labels = ['X', 'A', 'B', 'C', 'D']
        for point, label in zip(self.points, labels):
            ax.text(point.x, point.y, label,
                   color=self.color,
                   alpha=self.alpha)
                   
        # 添加比率标签
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            mid_x = (p1.x + p2.x) / 2
            mid_y = (p1.y + p2.y) / 2
            
            ratio_key = labels[i] + labels[i+1]
            if ratio_key in self.ratios:
                ratio = self.ratios[ratio_key]
                ax.text(mid_x, mid_y, f'{ratio:.3f}',
                       color=self.color,
                       alpha=self.alpha)
                       
    def contains(self, point: Point) -> bool:
        # 检查点是否在线段上
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            
            if dx == 0 and dy == 0:
                continue
                
            t = ((point.x - p1.x) * dx +
                 (point.y - p1.y) * dy) / (dx * dx + dy * dy)
                 
            if t < 0 or t > 1:
                continue
                
            px = p1.x + t * dx
            py = p1.y + t * dy
            
            dist = np.sqrt((point.x - px)**2 + (point.y - py)**2)
            if dist < 5:
                return True
                
        return False
        
    def move(self, dx: float, dy: float):
        for point in self.points:
            point.x += dx
            point.y += dy

class FibonacciRetracement(DrawingTool):
    """斐波那契回撤工具"""
    
    def __init__(self, start: Point, end: Point):
        super().__init__()
        self.start = start
        self.end = end
        self.levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
    def draw(self, ax):
        # 计算价格范围
        price_range = self.end.y - self.start.y
        x_range = [self.start.x, self.end.x]
        
        # 绘制回撤线
        for level in self.levels:
            y = self.start.y + price_range * (1 - level)
            ax.plot(x_range, [y, y],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
            
            # 添加标签
            ax.text(self.end.x, y,
                   f'{level*100:.1f}%',
                   color=self.color,
                   alpha=self.alpha)
                   
    def to_json(self) -> Dict[str, Any]:
        return {
            'type': 'FibonacciRetracement',
            'start': {'x': self.start.x, 'y': self.start.y},
            'end': {'x': self.end.x, 'y': self.end.y},
            'color': self.color,
            'line_style': self.line_style,
            'alpha': self.alpha
        }
        
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'FibonacciRetracement':
        tool = cls(
            Point(data['start']['x'], data['start']['y']),
            Point(data['end']['x'], data['end']['y'])
        )
        tool.color = data['color']
        tool.line_style = data['line_style']
        tool.alpha = data['alpha']
        return tool

class GannSquare(DrawingTool):
    """甘氏方格"""
    
    def __init__(self, center: Point, size: float, levels: int = 3):
        super().__init__()
        self.center = center
        self.size = size
        self.levels = levels
        
    def draw(self, ax):
        # 绘制水平和垂直线
        for i in range(-self.levels, self.levels + 1):
            # 水平线
            y = self.center.y + i * self.size
            ax.plot([self.center.x - self.levels * self.size,
                    self.center.x + self.levels * self.size],
                   [y, y],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
                   
            # 垂直线
            x = self.center.x + i * self.size
            ax.plot([x, x],
                   [self.center.y - self.levels * self.size,
                    self.center.y + self.levels * self.size],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
                   
        # 绘制对角线
        angles = [45, -45]
        for angle in angles:
            rad = np.radians(angle)
            dx = self.levels * self.size * np.cos(rad)
            dy = self.levels * self.size * np.sin(rad)
            
            ax.plot([self.center.x - dx, self.center.x + dx],
                   [self.center.y - dy, self.center.y + dy],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
                   
    def to_json(self) -> Dict[str, Any]:
        return {
            'type': 'GannSquare',
            'center': {'x': self.center.x, 'y': self.center.y},
            'size': self.size,
            'levels': self.levels,
            'color': self.color,
            'line_style': self.line_style,
            'alpha': self.alpha
        }
        
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'GannSquare':
        tool = cls(
            Point(data['center']['x'], data['center']['y']),
            data['size'],
            data['levels']
        )
        tool.color = data['color']
        tool.line_style = data['line_style']
        tool.alpha = data['alpha']
        return tool

class PitchforkExtension(DrawingTool):
    """叉子延伸工具"""
    
    def __init__(self, pivot: Point, p1: Point, p2: Point):
        super().__init__()
        self.pivot = pivot
        self.p1 = p1
        self.p2 = p2
        self.levels = [-2.0, -1.618, -1.0, -0.618, 0.0,
                      0.618, 1.0, 1.618, 2.0]
        
    def draw(self, ax):
        # 计算中线
        midpoint = Point((self.p1.x + self.p2.x) / 2,
                        (self.p1.y + self.p2.y) / 2)
                        
        # 计算叉子方向
        dx = midpoint.x - self.pivot.x
        dy = midpoint.y - self.pivot.y
        
        # 绘制中线
        ax.plot([self.pivot.x, midpoint.x + dx],
               [self.pivot.y, midpoint.y + dy],
               color=self.color,
               linestyle=self.line_style,
               alpha=self.alpha)
               
        # 绘制平行线
        for level in self.levels:
            offset_x = level * dx
            offset_y = level * dy
            
            ax.plot([self.p1.x + offset_x, self.p2.x + offset_x],
                   [self.p1.y + offset_y, self.p2.y + offset_y],
                   color=self.color,
                   linestyle=self.line_style,
                   alpha=self.alpha)
                   
            # 添加标签
            ax.text(self.p2.x + offset_x,
                   self.p2.y + offset_y,
                   f'{level:.3f}',
                   color=self.color,
                   alpha=self.alpha)
                   
    def to_json(self) -> Dict[str, Any]:
        return {
            'type': 'PitchforkExtension',
            'pivot': {'x': self.pivot.x, 'y': self.pivot.y},
            'p1': {'x': self.p1.x, 'y': self.p1.y},
            'p2': {'x': self.p2.x, 'y': self.p2.y},
            'color': self.color,
            'line_style': self.line_style,
            'alpha': self.alpha
        }
        
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'PitchforkExtension':
        tool = cls(
            Point(data['pivot']['x'], data['pivot']['y']),
            Point(data['p1']['x'], data['p1']['y']),
            Point(data['p2']['x'], data['p2']['y'])
        )
        tool.color = data['color']
        tool.line_style = data['line_style']
        tool.alpha = data['alpha']
        return tool

def save_tools(tools: List[DrawingTool], filename: str):
    """保存绘图工具"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(tools, f)
        
def load_tools(filename: str) -> List[DrawingTool]:
    """加载绘图工具"""
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_tools_json(tools: List[DrawingTool], filename: str):
    """将绘图工具保存为JSON格式"""
    import json
    
    tool_data = []
    for tool in tools:
        if hasattr(tool, 'to_json'):
            tool_data.append(tool.to_json())
            
    with open(filename, 'w') as f:
        json.dump(tool_data, f, indent=2)
        
def load_tools_json(filename: str) -> List[DrawingTool]:
    """从JSON文件加载绘图工具"""
    import json
    
    tool_classes = {
        'FibonacciRetracement': FibonacciRetracement,
        'GannSquare': GannSquare,
        'PitchforkExtension': PitchforkExtension,
        'WaveLine': WaveLine,
        'TimeCycle': TimeCycle,
        'ElliottWave': ElliottWave,
        'HarmonicPattern': HarmonicPattern
    }
    
    with open(filename, 'r') as f:
        tool_data = json.load(f)
        
    tools = []
    for data in tool_data:
        tool_class = tool_classes.get(data['type'])
        if tool_class and hasattr(tool_class, 'from_json'):
            tools.append(tool_class.from_json(data))
            
    return tools
