from typing import List, Optional, Generator
from dataclasses import dataclass
from enum import Enum, auto

class TokenType(Enum):
    """标记类型"""
    # 基础标记
    EOF = auto()  # 文件结束
    IDENTIFIER = auto()  # 标识符
    NUMBER = auto()  # 数字
    STRING = auto()  # 字符串
    COMMENT = auto()  # 注释
    
    # 基础关键字
    VERSION = auto()  # version
    STRATEGY = auto()  # strategy
    INDICATOR = auto()  # indicator
    PLOT = auto()  # plot
    VAR = auto()  # var
    CONST = auto()  # const
    FUNCTION = auto()  # function
    RETURN = auto()  # return
    EXPORT = auto()  # export
    IMPORT = auto()  # import
    STUDY = auto()  # study
    LIBRARY = auto()  # library
    
    # 控制流
    IF = auto()  # if
    ELSE = auto()  # else
    FOR = auto()  # for
    WHILE = auto()  # while
    BREAK = auto()  # break
    CONTINUE = auto()  # continue
    
    # 逻辑运算符
    AND = auto()  # and
    OR = auto()  # or
    NOT = auto()  # not
    XOR = auto()  # xor
    AND_AND = auto()  # &&
    OR_OR = auto()  # ||
    
    # 常量
    TRUE = auto()  # true
    FALSE = auto()  # false
    NULL = auto()  # null
    NA = auto()  # na
    
    # 变量声明和输入
    INPUT = auto()  # input
    
    # 数据类型
    FLOAT = auto()  # float
    INT = auto()  # int
    STRING_TYPE = auto()  # string
    BOOL = auto()  # bool
    COLOR = auto()  # color
    ARRAY_TYPE = auto()  # array
    MATRIX_TYPE = auto()  # matrix
    MAP_TYPE = auto()  # map
    
    # OHLCV数据
    OPEN = auto()  # open
    HIGH = auto()  # high
    LOW = auto()  # low
    CLOSE = auto()  # close
    VOLUME = auto()  # volume
    HL2 = auto()  # hl2
    HLC3 = auto()  # hlc3
    OHLC4 = auto()  # ohlc4
    
    # 技术分析函数
    TA_SMA = auto()  # Simple Moving Average
    TA_EMA = auto()  # Exponential Moving Average
    TA_RMA = auto()  # Running Moving Average
    TA_WMA = auto()  # Weighted Moving Average
    TA_VWMA = auto()  # Volume Weighted Moving Average
    TA_SWMA = auto()  # Symmetrically Weighted Moving Average
    TA_RSI = auto()  # Relative Strength Index
    TA_MACD = auto()  # Moving Average Convergence Divergence
    TA_BB = auto()  # Bollinger Bands
    TA_CCI = auto()  # Commodity Channel Index
    TA_ATR = auto()  # Average True Range
    TA_SUPERTREND = auto()  # Supertrend
    TA_STOCH = auto()  # Stochastic Oscillator
    TA_MOMENTUM = auto()  # Momentum
    TA_MFI = auto()  # Money Flow Index
    TA_VWAP = auto()  # Volume Weighted Average Price
    TA_CORRELATION = auto()  # Correlation
    TA_VARIANCE = auto()  # Variance
    TA_STDDEV = auto()  # Standard Deviation
    TA_PERCENTRANK = auto()  # Percentile Rank
    TA_PERCENTILE = auto()  # Percentile
    TA_HIGHEST = auto()  # Highest Value
    TA_LOWEST = auto()  # Lowest Value
    TA_AVERAGE = auto()  # Average
    TA_SUM = auto()  # Sum
    TA_CROSS = auto()  # Cross
    TA_CROSSOVER = auto()  # Crossover
    TA_CROSSUNDER = auto()  # Crossunder
    
    # 策略相关
    STRATEGY_LONG = auto()  # strategy.long
    STRATEGY_SHORT = auto()  # strategy.short
    STRATEGY_ENTRY = auto()  # strategy.entry
    STRATEGY_EXIT = auto()  # strategy.exit
    STRATEGY_CLOSE = auto()  # strategy.close
    STRATEGY_REVERSE = auto()  # strategy.reverse
    STRATEGY_PYRAMIDING = auto()  # strategy.pyramiding
    STRATEGY_RISK = auto()  # strategy.risk
    STRATEGY_COMMISSION = auto()  # strategy.commission
    STRATEGY_SLIPPAGE = auto()  # strategy.slippage
    STRATEGY_POSITION = auto()  # strategy.position
    STRATEGY_PROFIT = auto()  # strategy.profit
    
    # 绘图相关
    PLOT_LINE = auto()  # plot.line
    PLOT_HLINE = auto()  # plot.hline
    PLOT_ARROW = auto()  # plot.arrow
    PLOT_LABEL = auto()  # plot.label
    PLOT_BOX = auto()  # plot.box
    PLOT_TABLE = auto()  # plot.table
    PLOT_BGCOLOR = auto()  # plot.bgcolor
    PLOT_BARCOLOR = auto()  # plot.barcolor
    PLOT_FILL = auto()  # plot.fill
    
    # 颜色常量
    COLOR_RED = auto()  # color.red
    COLOR_GREEN = auto()  # color.green
    COLOR_BLUE = auto()  # color.blue
    COLOR_WHITE = auto()  # color.white
    COLOR_BLACK = auto()  # color.black
    COLOR_YELLOW = auto()  # color.yellow
    COLOR_PURPLE = auto()  # color.purple
    COLOR_ORANGE = auto()  # color.orange
    
    # 时间相关
    TIME = auto()  # time
    TIME_YEAR = auto()  # time.year
    TIME_MONTH = auto()  # time.month
    TIME_WEEKOFYEAR = auto()  # time.weekofyear
    TIME_DAYOFMONTH = auto()  # time.dayofmonth
    TIME_DAYOFWEEK = auto()  # time.dayofweek
    TIME_HOUR = auto()  # time.hour
    TIME_MINUTE = auto()  # time.minute
    TIME_SECOND = auto()  # time.second
    
    # 数学函数
    MATH_ABS = auto()  # abs
    MATH_POW = auto()  # pow
    MATH_SQRT = auto()  # sqrt
    MATH_LOG = auto()  # log
    MATH_EXP = auto()  # exp
    MATH_SIN = auto()  # sin
    MATH_COS = auto()  # cos
    MATH_TAN = auto()  # tan
    MATH_ROUND = auto()  # round
    MATH_FLOOR = auto()  # floor
    MATH_CEIL = auto()  # ceil
    MATH_MIN = auto()  # min
    MATH_MAX = auto()  # max
    
    # 字符串函数
    STR_LENGTH = auto()  # str.len
    STR_SUBSTRING = auto()  # str.substr
    STR_REPLACE = auto()  # str.replace
    STR_SPLIT = auto()  # str.split
    STR_JOIN = auto()  # str.join
    STR_FORMAT = auto()  # str.format
    
    # 数组和矩阵函数
    ARRAY_NEW = auto()  # array.new
    ARRAY_PUSH = auto()  # array.push
    ARRAY_POP = auto()  # array.pop
    ARRAY_GET = auto()  # array.get
    ARRAY_SET = auto()  # array.set
    ARRAY_SIZE = auto()  # array.size
    ARRAY_SLICE = auto()  # array.slice
    MATRIX_NEW = auto()  # matrix.new
    MATRIX_GET = auto()  # matrix.get
    MATRIX_SET = auto()  # matrix.set
    MATRIX_ROWS = auto()  # matrix.rows
    MATRIX_COLS = auto()  # matrix.cols
    
    # 运算符
    PLUS = auto()  # +
    MINUS = auto()  # -
    MULTIPLY = auto()  # *
    DIVIDE = auto()  # /
    MODULO = auto()  # %
    POWER = auto()  # **
    EQUALS = auto()  # =
    EQUALS_EQUALS = auto()  # ==
    NOT_EQUALS = auto()  # !=
    GREATER = auto()  # >
    GREATER_EQUALS = auto()  # >=
    LESS = auto()  # <
    LESS_EQUALS = auto()  # <=
    PLUS_EQUALS = auto()  # +=
    MINUS_EQUALS = auto()  # -=
    MULTIPLY_EQUALS = auto()  # *=
    DIVIDE_EQUALS = auto()  # /=
    MODULO_EQUALS = auto()  # %=
    QUESTION = auto()  # ?
    COLON = auto()  # :
    ARROW = auto()  # ->
    FAT_ARROW = auto()  # =>
    DOUBLE_COLON = auto()  # ::
    ELLIPSIS = auto()  # ...
    NULL_COALESCE = auto()  # ??
    BITWISE_AND = auto()  # &
    BITWISE_OR = auto()  # |
    BITWISE_XOR = auto()  # ^
    BITWISE_NOT = auto()  # ~
    LEFT_SHIFT = auto()  # <<
    RIGHT_SHIFT = auto()  # >>
    
    # 分隔符
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COMMA = auto()  # ,
    DOT = auto()  # .
    SEMICOLON = auto()  # ;
    AT = auto()  # @
    
    # 特殊标记
    VERSION_DECLARATION = auto()  # 版本声明

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if source else None

    def advance(self, step=1) -> str:
        """前进到下一个字符"""
        self.position += step
        self.column += step
        self.current_char = self.source[self.position] if self.position < len(self.source) else None
        return self.current_char

    def skip_whitespace(self) -> None:
        """跳过空白字符"""
        while self.current_char and self.current_char.isspace():
            if self.current_char == '\n':
                self.line += 1
                self.column = 0
            self.advance()

    def read_number(self) -> Token:
        """读取数字"""
        result = ''
        start_column = self.column
        
        # 处理前导符号
        if self.peek() in '+-':
            result += self.advance()
            self.column += 1

        # 处理整数部分
        while self.current_char and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        # 处理小数部分
        if self.current_char == '.':
            result += self.advance()  # 添加小数点
            self.column += 1
            
            # 至少需要一个小数位
            if self.current_char and self.current_char.isdigit():
                while self.current_char and self.current_char.isdigit():
                    result += self.current_char
                    self.advance()
            else:
                raise SyntaxError(f"Invalid decimal number at line {self.line}, column {self.column}")

        # 处理科学计数法
        if self.current_char and self.current_char.lower() == 'e':
            result += self.advance()  # 添加 'e'
            self.column += 1
            
            # 处理指数的符号
            if self.current_char in '+-':
                result += self.advance()
                self.column += 1
            
            # 指数部分必须是整数
            if self.current_char and self.current_char.isdigit():
                while self.current_char and self.current_char.isdigit():
                    result += self.current_char
                    self.advance()
            else:
                raise SyntaxError(f"Invalid scientific notation at line {self.line}, column {self.column}")

        return Token(TokenType.NUMBER, result, self.line, start_column)

    def read_string(self) -> Token:
        """读取字符串"""
        quote = self.advance()  # 获取引号类型
        self.column += 1
        
        value = ''
        while self.current_char and self.current_char != quote:
            if self.current_char == '\\':
                self.advance()  # 跳过反斜杠
                self.column += 1
                if self.current_char:
                    # 处理转义字符
                    escape_chars = {
                        'n': '\n',
                        't': '\t',
                        'r': '\r',
                        '"': '"',
                        "'": "'",
                        '\\': '\\'
                    }
                    next_char = self.advance()
                    self.column += 1
                    value += escape_chars.get(next_char, next_char)
            else:
                value += self.current_char
                self.advance()
                self.column += 1

        if not self.current_char:
            raise SyntaxError(f"Unterminated string at line {self.line}, column {self.column}")

        self.advance()  # 跳过结束引号
        self.column += 1

        return Token(TokenType.STRING, value, self.line, self.column - len(value) - 2)

    def read_identifier(self) -> Token:
        """读取标识符"""
        result = ''
        start_column = self.column
        
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_' or self.current_char == '.'):
            result += self.current_char
            self.advance()
            self.column += 1
            
        # 检查是否是关键字
        token_type = self.get_keyword_type(result)
        if token_type is None:
            token_type = TokenType.IDENTIFIER

        return Token(token_type, result, self.line, start_column)

    def read_comment(self) -> Token:
        """读取注释"""
        start_column = self.column
        comment = ''
        
        # 跳过开始的 '//'
        self.advance()  # 跳过第一个 '/'
        self.advance()  # 跳过第二个 '/'
        self.column += 2
        
        # 检查是否是特殊注释（如版本声明）
        if not self.is_at_end() and self.peek() == '@':
            self.advance()  # 跳过 '@'
            self.column += 1
            
            # 读取特殊标记
            tag = ''
            while not self.is_at_end() and self.peek().isalpha():
                tag += self.advance()
                self.column += 1
                
            if tag == 'version':
                # 处理版本声明
                while not self.is_at_end() and self.peek().isspace():
                    self.advance()
                    self.column += 1
                    
                if self.peek() == '=':
                    self.advance()  # 跳过 '='
                    self.column += 1
                    
                    # 读取版本号
                    version = ''
                    while not self.is_at_end() and (self.peek().isdigit() or self.peek() == '.'):
                        version += self.advance()
                        self.column += 1
                        
                    return Token(TokenType.VERSION_DECLARATION, f'@version={version}', self.line, start_column)
            
            # 其他特殊标记可以在这里添加处理
            
        # 处理普通注释
        while not self.is_at_end() and self.peek() != '\n':
            comment += self.advance()
            self.column += 1
            
        return Token(TokenType.COMMENT, '//' + comment, self.line, start_column)

    def try_match_operator(self) -> Optional[Token]:
        """尝试匹配操作符"""
        start_column = self.column
        
        # 定义多字符操作符
        operators = {
            # 赋值操作符
            '=': TokenType.EQUALS,
            '+=': TokenType.PLUS_EQUALS,
            '-=': TokenType.MINUS_EQUALS,
            '*=': TokenType.MULTIPLY_EQUALS,
            '/=': TokenType.DIVIDE_EQUALS,
            '%=': TokenType.MODULO_EQUALS,
            
            # 算术操作符
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO,
            '**': TokenType.POWER,
            
            # 比较操作符
            '==': TokenType.EQUALS_EQUALS,
            '!=': TokenType.NOT_EQUALS,
            '>': TokenType.GREATER,
            '>=': TokenType.GREATER_EQUALS,
            '<': TokenType.LESS,
            '<=': TokenType.LESS_EQUALS,
            '<>': TokenType.NOT_EQUALS,  # 兼容性支持
            
            # 逻辑操作符
            '&&': TokenType.AND_AND,
            '||': TokenType.OR_OR,
            '!': TokenType.NOT,
            
            # 位运算操作符
            '&': TokenType.BITWISE_AND,
            '|': TokenType.BITWISE_OR,
            '^': TokenType.BITWISE_XOR,
            '~': TokenType.BITWISE_NOT,
            '<<': TokenType.LEFT_SHIFT,
            '>>': TokenType.RIGHT_SHIFT,
            
            # 其他操作符
            '->': TokenType.ARROW,
            '=>': TokenType.FAT_ARROW,
            '::': TokenType.DOUBLE_COLON,
            '...': TokenType.ELLIPSIS,
            '??': TokenType.NULL_COALESCE,
        }
        
        # 尝试匹配最长的操作符
        current = ''
        matched_operator = None
        matched_type = None
        
        while not self.is_at_end():
            current += self.peek()
            if current in operators:
                matched_operator = current
                matched_type = operators[current]
                self.advance()
                self.column += 1
            else:
                # 如果不能继续匹配，就停止
                break
                
        if matched_operator:
            return Token(matched_type, matched_operator, self.line, start_column)
        return None

    def get_next_token(self) -> Token:
        """获取下一个标记"""
        while self.current_char:
            # 跳过空白字符
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # 注释
            if self.current_char == '/' and (self.peek_next() == '/' or self.peek_next() == '*'):
                return self.read_comment()

            # 字符串
            if self.current_char in ['"', "'"]:
                return self.read_string()

            # 数字
            if self.current_char.isdigit():
                return self.read_number()

            # 标识符和关键字
            if self.current_char.isalpha() or self.current_char == '_':
                return self.read_identifier()

            # 操作符
            if self.is_operator_start(self.current_char):
                return self.try_match_operator()

            # 其他标记
            if self.current_char == '(':
                self.advance()
                self.column += 1
                return Token(TokenType.LPAREN, '(', self.line, self.column - 1)
            elif self.current_char == ')':
                self.advance()
                self.column += 1
                return Token(TokenType.RPAREN, ')', self.line, self.column - 1)
            elif self.current_char == '{':
                self.advance()
                self.column += 1
                return Token(TokenType.LBRACE, '{', self.line, self.column - 1)
            elif self.current_char == '}':
                self.advance()
                self.column += 1
                return Token(TokenType.RBRACE, '}', self.line, self.column - 1)
            elif self.current_char == ',':
                self.advance()
                self.column += 1
                return Token(TokenType.COMMA, ',', self.line, self.column - 1)
            elif self.current_char == '.':
                self.advance()
                self.column += 1
                return Token(TokenType.DOT, '.', self.line, self.column - 1)
            elif self.current_char == '@':
                self.advance()
                self.column += 1
                return Token(TokenType.AT, '@', self.line, self.column - 1)

            raise SyntaxError(f"未知字符 '{self.current_char}' 在第 {self.line} 行，第 {self.column} 列")

        return Token(TokenType.EOF, '', self.line, self.column)

    def peek(self) -> str:
        """查看当前字符"""
        if self.is_at_end():
            return '\0'
        return self.source[self.position]

    def peek_next(self, offset: int = 1) -> str:
        """查看后面的字符"""
        if self.position + offset >= len(self.source):
            return '\0'
        return self.source[self.position + offset]

    def advance(self, count: int = 1) -> str:
        """向前移动指定数量的字符"""
        char = self.peek()
        self.position += count
        return char

    def is_at_end(self) -> bool:
        """检查是否到达源代码末尾"""
        return self.position >= len(self.source)

    def is_operator_start(self, char: str) -> bool:
        """检查字符是否可能是操作符的开始"""
        operator_starts = '=+-*/%!&|^~<>?.'
        return char in operator_starts

    def get_keyword_type(self, identifier: str) -> Optional[TokenType]:
        """获取关键字对应的标记类型"""
        keywords = {
            'version': TokenType.VERSION,
            'strategy': TokenType.STRATEGY,
            'indicator': TokenType.INDICATOR,
            'plot': TokenType.PLOT,
            'var': TokenType.VAR,
            'const': TokenType.CONST,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'for': TokenType.FOR,
            'while': TokenType.WHILE,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
            'return': TokenType.RETURN,
            'function': TokenType.FUNCTION,
            
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'xor': TokenType.XOR,
            
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
            'null': TokenType.NULL,
            'na': TokenType.NA,
            
            'input': TokenType.INPUT,
            'export': TokenType.EXPORT,
            'import': TokenType.IMPORT,
            'study': TokenType.STUDY,
            'library': TokenType.LIBRARY,
            
            'float': TokenType.FLOAT,
            'int': TokenType.INT,
            'string': TokenType.STRING_TYPE,
            'bool': TokenType.BOOL,
            'color': TokenType.COLOR,
            'array': TokenType.ARRAY_TYPE,
            'matrix': TokenType.MATRIX_TYPE,
            'map': TokenType.MAP_TYPE,
            
            'open': TokenType.OPEN,
            'high': TokenType.HIGH,
            'low': TokenType.LOW,
            'close': TokenType.CLOSE,
            'volume': TokenType.VOLUME,
            'time': TokenType.TIME,
            'hl2': TokenType.HL2,
            'hlc3': TokenType.HLC3,
            'ohlc4': TokenType.OHLC4,
            
            'sma': TokenType.TA_SMA,
            'ema': TokenType.TA_EMA,
            'rma': TokenType.TA_RMA,
            'wma': TokenType.TA_WMA,
            'vwma': TokenType.TA_VWMA,
            'swma': TokenType.TA_SWMA,
            'rsi': TokenType.TA_RSI,
            'macd': TokenType.TA_MACD,
            'bb': TokenType.TA_BB,
            'cci': TokenType.TA_CCI,
            'atr': TokenType.TA_ATR,
            'supertrend': TokenType.TA_SUPERTREND,
            'stoch': TokenType.TA_STOCH,
            'mom': TokenType.TA_MOMENTUM,
            'mfi': TokenType.TA_MFI,
            'vwap': TokenType.TA_VWAP,
            'correlation': TokenType.TA_CORRELATION,
            'variance': TokenType.TA_VARIANCE,
            'stdev': TokenType.TA_STDDEV,
            'percentrank': TokenType.TA_PERCENTRANK,
            'percentile': TokenType.TA_PERCENTILE,
            'highest': TokenType.TA_HIGHEST,
            'lowest': TokenType.TA_LOWEST,
            'avg': TokenType.TA_AVERAGE,
            'sum': TokenType.TA_SUM,
            'cross': TokenType.TA_CROSS,
            'crossover': TokenType.TA_CROSSOVER,
            'crossunder': TokenType.TA_CROSSUNDER,
            
            'strategy': TokenType.STRATEGY,
            'long': TokenType.STRATEGY_LONG,
            'short': TokenType.STRATEGY_SHORT,
            'entry': TokenType.STRATEGY_ENTRY,
            'exit': TokenType.STRATEGY_EXIT,
            'close': TokenType.STRATEGY_CLOSE,
            'reverse': TokenType.STRATEGY_REVERSE,
            'pyramiding': TokenType.STRATEGY_PYRAMIDING,
            'risk': TokenType.STRATEGY_RISK,
            'commission': TokenType.STRATEGY_COMMISSION,
            'slippage': TokenType.STRATEGY_SLIPPAGE,
            'position': TokenType.STRATEGY_POSITION,
            'profit': TokenType.STRATEGY_PROFIT,
            
            'plot': TokenType.PLOT,
            'hline': TokenType.PLOT_HLINE,
            'line': TokenType.PLOT_LINE,
            'label': TokenType.PLOT_LABEL,
            'box': TokenType.PLOT_BOX,
            'table': TokenType.PLOT_TABLE,
            'bgcolor': TokenType.PLOT_BGCOLOR,
            'barcolor': TokenType.PLOT_BARCOLOR,
            'fill': TokenType.PLOT_FILL,
            
            'color.red': TokenType.COLOR_RED,
            'color.green': TokenType.COLOR_GREEN,
            'color.blue': TokenType.COLOR_BLUE,
            'color.white': TokenType.COLOR_WHITE,
            'color.black': TokenType.COLOR_BLACK,
            'color.yellow': TokenType.COLOR_YELLOW,
            'color.purple': TokenType.COLOR_PURPLE,
            'color.orange': TokenType.COLOR_ORANGE,
            
            'time': TokenType.TIME,
            'year': TokenType.TIME_YEAR,
            'month': TokenType.TIME_MONTH,
            'weekofyear': TokenType.TIME_WEEKOFYEAR,
            'dayofmonth': TokenType.TIME_DAYOFMONTH,
            'dayofweek': TokenType.TIME_DAYOFWEEK,
            'hour': TokenType.TIME_HOUR,
            'minute': TokenType.TIME_MINUTE,
            'second': TokenType.TIME_SECOND,
            
            'abs': TokenType.MATH_ABS,
            'pow': TokenType.MATH_POW,
            'sqrt': TokenType.MATH_SQRT,
            'log': TokenType.MATH_LOG,
            'exp': TokenType.MATH_EXP,
            'sin': TokenType.MATH_SIN,
            'cos': TokenType.MATH_COS,
            'tan': TokenType.MATH_TAN,
            'round': TokenType.MATH_ROUND,
            'floor': TokenType.MATH_FLOOR,
            'ceil': TokenType.MATH_CEIL,
            'min': TokenType.MATH_MIN,
            'max': TokenType.MATH_MAX,
            
            'str.len': TokenType.STR_LENGTH,
            'str.substr': TokenType.STR_SUBSTRING,
            'str.replace': TokenType.STR_REPLACE,
            'str.split': TokenType.STR_SPLIT,
            'str.join': TokenType.STR_JOIN,
            'str.format': TokenType.STR_FORMAT,
            
            'array.new': TokenType.ARRAY_NEW,
            'array.push': TokenType.ARRAY_PUSH,
            'array.pop': TokenType.ARRAY_POP,
            'array.get': TokenType.ARRAY_GET,
            'array.set': TokenType.ARRAY_SET,
            'array.size': TokenType.ARRAY_SIZE,
            'array.slice': TokenType.ARRAY_SLICE,
            'matrix.new': TokenType.MATRIX_NEW,
            'matrix.get': TokenType.MATRIX_GET,
            'matrix.set': TokenType.MATRIX_SET,
            'matrix.rows': TokenType.MATRIX_ROWS,
            'matrix.cols': TokenType.MATRIX_COLS,
        }
        
        return keywords.get(identifier.lower())

    def tokenize(self) -> Generator[Token, None, None]:
        """将源代码转换为标记流"""
        self.position = 0
        self.line = 1
        self.column = 1

        while not self.is_at_end():
            # 跳过空白字符
            while not self.is_at_end() and self.peek().isspace():
                if self.peek() == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.advance()

            if self.is_at_end():
                break

            char = self.peek()

            # 处理注释
            if char == '/' and self.peek_next() == '/':
                token = self.read_comment()
                if token:
                    yield token
                continue

            # 处理数字
            if char.isdigit() or (char == '.' and self.peek_next().isdigit()):
                token = self.read_number()
                if token:
                    yield token
                continue

            # 处理标识符和关键字
            if char.isalpha() or char == '_':
                token = self.read_identifier()
                if token:
                    yield token
                continue

            # 处理字符串
            if char == '"' or char == "'":
                token = self.read_string()
                if token:
                    yield token
                continue

            # 处理操作符和分隔符
            if self.is_operator_start(char):
                token = self.try_match_operator()
                if token:
                    yield token
                    continue

            # 处理其他单字符标记
            if char in '(){}[],.;?:':
                token = self.read_single_char()
                if token:
                    yield token
                continue

            raise SyntaxError(f"Unexpected character: {char} at line {self.line}, column {self.column}")

    def read_single_char(self) -> Optional[Token]:
        """读取单字符标记"""
        char = self.advance()
        self.column += 1

        token_types = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
            ';': TokenType.SEMICOLON,
            '?': TokenType.QUESTION,
            ':': TokenType.COLON
        }

        token_type = token_types.get(char)
        if token_type:
            return Token(token_type, char, self.line, self.column - 1)
        return None
