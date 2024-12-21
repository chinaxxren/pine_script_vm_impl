from typing import List, Optional, Any
from dataclasses import dataclass
from enum import Enum, auto
from .lexer import Token, TokenType

class ASTNodeType(Enum):
    PROGRAM = auto()
    VERSION_DECLARATION = auto()
    INDICATOR_DECL = auto()
    PLOT_STMT = auto()
    BINARY_OP = auto()
    NUMBER = auto()
    IDENTIFIER = auto()
    ASSIGNMENT = auto()
    FUNCTION_CALL = auto()
    IF_STMT = auto()
    WHILE_STMT = auto()
    FOR_STMT = auto()
    BREAK_STMT = auto()
    CONTINUE_STMT = auto()
    BLOCK = auto()
    COMPARISON = auto()
    LOGICAL_OP = auto()
    INDICATOR_CALL = auto()
    INDICATOR_PARAM = auto()
    STRATEGY_ENTRY = auto()
    STRATEGY_EXIT = auto()
    STRATEGY_CLOSE = auto()
    STRATEGY_DECL = auto()  # 添加策略声明节点类型
    COMMENT = auto()  # 添加注释节点类型
    VAR_DECL = auto()  # 添加变量声明节点类型
    TYPE = auto()  # 添加类型节点类型
    STRING = auto()  # 添加字符串字面量节点类型
    STRATEGY_DIRECTION = auto()  # 添加策略方向节点类型
    PRICE_SOURCE = auto()  # 添加价格来源节点类型
    INPUT_FUNCTION = auto()  # 添加输入函数节点类型
    TA_FUNCTION = auto()  # 添加技术分析函数节点类型
    STRATEGY_STATEMENT = auto()  # 添加策略语句节点类型
    VAR_DECLARATION = auto()  # 添加变量声明节点类型
    FUNCTION_DEFINITION = auto()  # 添加函数定义节点类型
    PARAMETER = auto()  # 添加参数节点类型
    PARAMETERS = auto()  # 添加参数列表节点类型
    RETURN_TYPE = auto()  # 添加返回类型节点类型
    RETURN_STMT = auto()  # 添加返回语句节点类型
    CONST_DECLARATION = auto()  # 添加常量声明节点类型
    ARRAY_METHOD = auto()  # 添加数组方法调用节点类型
    ARRAY_LITERAL = auto()  # 添加数组字面量节点类型
    NAMED_ARGUMENT = auto()  # 添加命名参数节点类型
    PLOT_STATEMENT = auto()  # 添加绘图语句节点类型
    TERNARY = auto()  # 添加三元表达式节点类型

@dataclass
class ASTNode:
    type: ASTNodeType
    value: Any = None
    children: List['ASTNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

class ParseError(Exception):
    """解析错误异常"""
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0

    def peek(self) -> Optional[Token]:
        """查看当前标记"""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def peek_next(self) -> Optional[Token]:
        """查看下一个标记"""
        if self.position + 1 < len(self.tokens):
            return self.tokens[self.position + 1]
        return None

    def advance(self) -> Token:
        """移动到下一个标记"""
        token = self.peek()
        if token is None:
            raise ParseError("Unexpected end of input")
        self.position += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """期望下一个标记是指定类型"""
        token = self.peek()
        if token is None or token.type != token_type:
            raise ParseError(f"Expected {token_type}, got {token.type if token else 'EOF'}")
        return self.advance()

    def consume(self, token_type: TokenType, message: str) -> Token:
        """消耗并返回当前标记，如果类型不匹配则抛出异常"""
        if self.check(token_type):
            return self.advance()
        raise ParseError(message)

    def check(self, token_type: TokenType) -> bool:
        """检查当前标记是否为指定类型"""
        if self.is_at_end():
            return False
        return self.peek().type == token_type

    def is_at_end(self) -> bool:
        """是否到达标记列表末尾"""
        return self.position >= len(self.tokens)

    def parse(self) -> ASTNode:
        """解析完整的Pine Script程序"""
        program = ASTNode(ASTNodeType.PROGRAM)
        
        while self.peek() and self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.VERSION:
                program.children.append(self.parse_version_declaration())
            elif self.peek().type == TokenType.STRATEGY:
                program.children.append(self.parse_strategy_declaration())
            elif self.peek().type == TokenType.INDICATOR:
                program.children.append(self.parse_indicator_declaration())
            elif self.peek().type == TokenType.PLOT:
                program.children.append(self.parse_plot_statement())
            elif self.peek().type == TokenType.COMMENT:
                program.children.append(self.parse_comment())
            else:
                program.children.append(self.parse_statement())
                
        return program

    def parse_version_declaration(self) -> ASTNode:
        """解析版本声明，如 //@version=5"""
        # 跳过 '//'
        self.advance()  # COMMENT
        self.advance()  # AT
        self.expect(TokenType.VERSION)
        self.expect(TokenType.EQUAL)
        version = self.expect(TokenType.NUMBER)
        return ASTNode(ASTNodeType.VERSION_DECLARATION, value=version.value)

    def parse_strategy_declaration(self) -> ASTNode:
        """解析策略声明，如 strategy("Test Strategy")"""
        self.expect(TokenType.STRATEGY)
        self.expect(TokenType.LPAREN)
        name = self.expect(TokenType.STRING)
        self.expect(TokenType.RPAREN)
        return ASTNode(ASTNodeType.STRATEGY_DECL, value=name.value)

    def parse_indicator_declaration(self) -> ASTNode:
        """解析指标声明，如 indicator("My Indicator")"""
        self.expect(TokenType.INDICATOR)
        self.expect(TokenType.LPAREN)
        name = self.expect(TokenType.STRING)
        self.expect(TokenType.RPAREN)
        return ASTNode(ASTNodeType.INDICATOR_DECL, value=name.value)

    def parse_plot_statement(self) -> ASTNode:
        """解析绘图语句，如 plot(close)"""
        self.expect(TokenType.PLOT)
        self.expect(TokenType.LPAREN)
        expr = self.parse_expression()
        self.expect(TokenType.RPAREN)
        return ASTNode(ASTNodeType.PLOT_STMT, children=[expr])

    def parse_comment(self) -> ASTNode:
        """解析注释"""
        token = self.advance()
        return ASTNode(ASTNodeType.COMMENT, value=token.value)

    def parse_statement(self) -> ASTNode:
        """解析语句"""
        token = self.peek()

        # 处理函数定义
        if token.type == TokenType.FUNCTION:
            return self.parse_function_definition()

        # 处理返回语句
        if token.type == TokenType.RETURN:
            return self.parse_return_statement()

        # 处理常量声明
        if token.type == TokenType.CONST:
            return self.parse_const_declaration()

        # 处理数组操作
        if token.type == TokenType.ARRAY:
            return self.parse_array_operation()

        # 处理绘图语句
        if token.type in [TokenType.PLOT_LINE, TokenType.PLOT_ARROW, TokenType.PLOT_SHAPE, 
                         TokenType.PLOT_LABEL, TokenType.PLOT_BGCOLOR, TokenType.PLOT_BARCOLOR]:
            return self.parse_plot_statement()

        # 处理策略语句
        if token.type in [TokenType.STRATEGY_ENTRY, TokenType.STRATEGY_EXIT, TokenType.STRATEGY_CLOSE,
                         TokenType.STRATEGY_CANCEL, TokenType.STRATEGY_RISK]:
            return self.parse_strategy_statement()

        # 处理技术分析函数
        if token.type in [TokenType.TA_SMA, TokenType.TA_EMA, TokenType.TA_RSI, TokenType.TA_MACD,
                         TokenType.TA_CROSSOVER, TokenType.TA_CROSSUNDER, TokenType.TA_HIGHEST,
                         TokenType.TA_LOWEST, TokenType.TA_BARSSINCE, TokenType.TA_CORRELATION,
                         TokenType.TA_VARIANCE, TokenType.TA_STDDEV]:
            return self.parse_ta_function()

        # 处理条件语句
        if token.type == TokenType.IF:
            return self.parse_if_statement()

        # 处理循环语句
        if token.type == TokenType.FOR:
            return self.parse_for_statement()

        if token.type == TokenType.WHILE:
            return self.parse_while_statement()

        return self.parse_expression()

    def parse_expression(self) -> ASTNode:
        """解析表达式"""
        return self.parse_ternary()

    def parse_ternary(self) -> ASTNode:
        """解析三元表达式"""
        condition = self.parse_logical_or()

        if self.peek() and self.peek().type == TokenType.QUESTION:
            self.advance()  # 跳过 '?'
            then_branch = self.parse_expression()
            self.expect(TokenType.COLON)
            else_branch = self.parse_expression()
            return ASTNode(ASTNodeType.TERNARY, children=[condition, then_branch, else_branch])

        return condition

    def parse_logical_or(self) -> ASTNode:
        """解析逻辑或表达式"""
        left = self.parse_logical_and()

        while self.peek() and self.peek().type == TokenType.OR:
            operator = self.advance()
            right = self.parse_logical_and()
            left = ASTNode(ASTNodeType.LOGICAL_OP, value=operator.value, children=[left, right])

        return left

    def parse_logical_and(self) -> ASTNode:
        """解析逻辑与表达式"""
        left = self.parse_comparison()

        while self.peek() and self.peek().type == TokenType.AND:
            operator = self.advance()
            right = self.parse_comparison()
            left = ASTNode(ASTNodeType.LOGICAL_OP, value=operator.value, children=[left, right])

        return left

    def parse_comparison(self) -> ASTNode:
        """解析比较表达式"""
        left = self.parse_additive()

        comparison_ops = [
            TokenType.EQUALS_EQUALS,
            TokenType.NOT_EQUALS,
            TokenType.GREATER,
            TokenType.GREATER_EQUALS,
            TokenType.LESS,
            TokenType.LESS_EQUALS
        ]

        while self.peek() and self.peek().type in comparison_ops:
            operator = self.advance()
            right = self.parse_additive()
            left = ASTNode(ASTNodeType.COMPARISON, value=operator.value, children=[left, right])

        return left

    def parse_additive(self) -> ASTNode:
        """解析加减表达式"""
        left = self.parse_multiplicative()

        while self.peek() and self.peek().type in [TokenType.PLUS, TokenType.MINUS]:
            operator = self.advance()
            right = self.parse_multiplicative()
            left = ASTNode(ASTNodeType.BINARY_OP, value=operator.value, children=[left, right])

        return left

    def parse_multiplicative(self) -> ASTNode:
        """解析乘除表达式"""
        left = self.parse_unary()

        while self.peek() and self.peek().type in [TokenType.MULTIPLY, TokenType.DIVIDE]:
            operator = self.advance()
            right = self.parse_unary()
            left = ASTNode(ASTNodeType.BINARY_OP, value=operator.value, children=[left, right])

        return left

    def parse_unary(self) -> ASTNode:
        """解析一元表达式"""
        if self.peek() and self.peek().type in [TokenType.PLUS, TokenType.MINUS, TokenType.NOT]:
            operator = self.advance()
            operand = self.parse_unary()
            return ASTNode(ASTNodeType.BINARY_OP, value=operator.value, children=[
                ASTNode(ASTNodeType.NUMBER, value=0),  # 对于负数，转换为 0 - operand
                operand
            ])
        return self.parse_factor()

    def parse_factor(self) -> ASTNode:
        """解析因子（数字、标识符、括号表达式）"""
        token = self.peek()

        if token.type == TokenType.NUMBER:
            self.advance()
            if token.value:  # 确保数字值不为空
                return ASTNode(ASTNodeType.NUMBER, value=float(token.value))
            else:
                raise ParseError("Empty number value")

        elif token.type == TokenType.STRING:
            self.advance()
            return ASTNode(ASTNodeType.STRING, value=token.value)

        elif token.type == TokenType.IDENTIFIER:
            identifier = self.advance()
            # 检查是否是函数调用
            if self.peek() and self.peek().type == TokenType.LPAREN:
                self.advance()  # 跳过 '('
                args = []
                while not self.is_at_end() and self.peek().type != TokenType.RPAREN:
                    args.append(self.parse_expression())
                    if self.peek().type == TokenType.COMMA:
                        self.advance()  # 跳过 ','
                if not self.is_at_end():
                    self.advance()  # 跳过 ')'
                return ASTNode(ASTNodeType.FUNCTION_CALL, value=identifier.value, children=args)
            return ASTNode(ASTNodeType.IDENTIFIER, value=identifier.value)

        elif token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        elif token.type == TokenType.COMMENT:
            return self.parse_comment()

        elif token.type == TokenType.INPUT:
            return self.parse_input_function()

        elif token.type in [TokenType.TA_SMA, TokenType.TA_RSI, TokenType.TA_CROSSOVER, TokenType.TA_CROSSUNDER]:
            return self.parse_ta_function()

        elif token.type in [TokenType.STRATEGY_LONG, TokenType.STRATEGY_SHORT]:
            self.advance()
            return ASTNode(ASTNodeType.STRATEGY_DIRECTION, value=token.value)

        elif token.type in [TokenType.CLOSE, TokenType.OPEN, TokenType.HIGH, TokenType.LOW, TokenType.VOLUME]:
            self.advance()
            return ASTNode(ASTNodeType.PRICE_SOURCE, value=token.value)

        elif token.type == TokenType.EQUALS:
            self.advance()  # 跳过 '='
            return self.parse_expression()

        raise SyntaxError(f"Unexpected token: {token.type}")

    def parse_function_definition(self) -> ASTNode:
        """解析函数定义"""
        self.advance()  # 跳过 'function' 关键字
        name = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LPAREN)
        
        # 解析参数列表
        params = []
        while not self.is_at_end() and self.peek().type != TokenType.RPAREN:
            if self.peek().type == TokenType.IDENTIFIER:
                param_name = self.advance()
                param_type = None
                if self.peek().type == TokenType.COLON:
                    self.advance()  # 跳过 ':'
                    param_type = self.advance()  # 获取类型
                params.append(ASTNode(ASTNodeType.PARAMETER, value=param_name.value, 
                                   children=[ASTNode(ASTNodeType.TYPE, value=param_type.value) if param_type else None]))
            
            if self.peek().type == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RPAREN)
        
        # 解析返回类型（如果有）
        return_type = None
        if self.peek().type == TokenType.COLON:
            self.advance()  # 跳过 ':'
            return_type = self.advance()

        # 解析函数体
        self.expect(TokenType.LBRACE)
        body = self.parse_block()
        self.expect(TokenType.RBRACE)

        return ASTNode(ASTNodeType.FUNCTION_DEFINITION, value=name.value, children=[
            ASTNode(ASTNodeType.PARAMETERS, children=params),
            ASTNode(ASTNodeType.RETURN_TYPE, value=return_type.value if return_type else None),
            body
        ])

    def parse_return_statement(self) -> ASTNode:
        """解析返回语句"""
        self.advance()  # 跳过 'return'
        if self.peek().type == TokenType.SEMICOLON:
            self.advance()
            return ASTNode(ASTNodeType.RETURN_STMT)
        value = self.parse_expression()
        if self.peek().type == TokenType.SEMICOLON:
            self.advance()
        return ASTNode(ASTNodeType.RETURN_STMT, children=[value])

    def parse_const_declaration(self) -> ASTNode:
        """解析常量声明"""
        self.advance()  # 跳过 'const'
        name = self.expect(TokenType.IDENTIFIER)
        
        # 解析类型（如果有）
        var_type = None
        if self.peek().type == TokenType.COLON:
            self.advance()  # 跳过 ':'
            var_type = self.advance()

        self.expect(TokenType.EQUALS)
        value = self.parse_expression()
        
        if self.peek().type == TokenType.SEMICOLON:
            self.advance()

        return ASTNode(ASTNodeType.CONST_DECLARATION, value=name.value, children=[
            ASTNode(ASTNodeType.TYPE, value=var_type.value if var_type else None),
            value
        ])

    def parse_array_operation(self) -> ASTNode:
        """解析数组操作"""
        self.advance()  # 跳过 'array'
        
        if self.peek().type == TokenType.DOT:
            self.advance()  # 跳过 '.'
            method = self.advance()  # 获取方法名
            
            self.expect(TokenType.LPAREN)
            args = []
            while not self.is_at_end() and self.peek().type != TokenType.RPAREN:
                args.append(self.parse_expression())
                if self.peek().type == TokenType.COMMA:
                    self.advance()
            self.expect(TokenType.RPAREN)
            
            return ASTNode(ASTNodeType.ARRAY_METHOD, value=method.value, children=args)
        
        # 数组字面量
        self.expect(TokenType.LBRACKET)
        elements = []
        while not self.is_at_end() and self.peek().type != TokenType.RBRACKET:
            elements.append(self.parse_expression())
            if self.peek().type == TokenType.COMMA:
                self.advance()
        self.expect(TokenType.RBRACKET)
        
        return ASTNode(ASTNodeType.ARRAY_LITERAL, children=elements)

    def parse_plot_statement(self) -> ASTNode:
        """解析绘图语句"""
        plot_type = self.advance()
        self.expect(TokenType.LPAREN)
        
        args = []
        while not self.is_at_end() and self.peek().type != TokenType.RPAREN:
            if self.peek().type == TokenType.IDENTIFIER:
                arg_name = self.advance()
                if self.peek().type == TokenType.EQUALS:
                    self.advance()  # 跳过 '='
                    arg_value = self.parse_expression()
                    args.append(ASTNode(ASTNodeType.NAMED_ARGUMENT, value=arg_name.value, children=[arg_value]))
                else:
                    self.position -= 1  # 回退
                    args.append(self.parse_expression())
            else:
                args.append(self.parse_expression())
            
            if self.peek().type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RPAREN)
        
        return ASTNode(ASTNodeType.PLOT_STATEMENT, value=plot_type.value, children=args)

    def parse_strategy_statement(self) -> ASTNode:
        """解析策略语句"""
        strategy_type = self.advance()
        self.expect(TokenType.LPAREN)
        
        args = []
        while not self.is_at_end() and self.peek().type != TokenType.RPAREN:
            if self.peek().type == TokenType.IDENTIFIER:
                arg_name = self.advance()
                if self.peek().type == TokenType.EQUALS:
                    self.advance()  # 跳过 '='
                    arg_value = self.parse_expression()
                    args.append(ASTNode(ASTNodeType.NAMED_ARGUMENT, value=arg_name.value, children=[arg_value]))
                else:
                    self.position -= 1  # 回退
                    args.append(self.parse_expression())
            else:
                args.append(self.parse_expression())
            
            if self.peek().type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RPAREN)
        
        return ASTNode(ASTNodeType.STRATEGY_STATEMENT, value=strategy_type.value, children=args)

    def parse_comment(self) -> ASTNode:
        """解析注释"""
        token = self.expect(TokenType.COMMENT)
        return ASTNode(ASTNodeType.COMMENT, value=token.value)

    def parse_var_declaration(self) -> ASTNode:
        """解析变量声明"""
        self.expect(TokenType.VAR)
        var_type = self.expect(TokenType.FLOAT)
        identifier = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.EQUALS)
        value = self.parse_expression()
        return ASTNode(ASTNodeType.VAR_DECLARATION, value=identifier.value, children=[
            ASTNode(ASTNodeType.TYPE, value=var_type.value),
            value
        ])

    def parse_input_function(self) -> ASTNode:
        """解析输入函数"""
        self.expect(TokenType.INPUT)
        self.expect(TokenType.LPAREN)
        args = []
        while not self.is_at_end() and self.peek().type != TokenType.RPAREN:
            args.append(self.parse_expression())
            if self.peek().type == TokenType.COMMA:
                self.advance()
        self.expect(TokenType.RPAREN)
        return ASTNode(ASTNodeType.INPUT_FUNCTION, value='input', children=args)

    def parse_ta_function(self) -> ASTNode:
        """解析技术分析函数"""
        token = self.advance()
        self.expect(TokenType.LPAREN)
        args = []
        while not self.is_at_end() and self.peek().type != TokenType.RPAREN:
            args.append(self.parse_expression())
            if self.peek().type == TokenType.COMMA:
                self.advance()
        self.expect(TokenType.RPAREN)
        return ASTNode(ASTNodeType.TA_FUNCTION, value=token.value, children=args)

    def parse_strategy_statement(self) -> ASTNode:
        """解析策略语句"""
        token = self.advance()
        self.expect(TokenType.LPAREN)
        args = []
        while not self.is_at_end() and self.peek().type != TokenType.RPAREN:
            args.append(self.parse_expression())
            if self.peek().type == TokenType.COMMA:
                self.advance()
        self.expect(TokenType.RPAREN)
        return ASTNode(ASTNodeType.STRATEGY_STATEMENT, value=token.value, children=args)

    def parse_if_statement(self) -> ASTNode:
        """解析if语句"""
        self.advance()  # 跳过 'if'
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)

        # 解析then分支
        then_branch = None
        if self.peek().type == TokenType.LBRACE:
            self.advance()  # 跳过 '{'
            then_branch = self.parse_block()
            self.expect(TokenType.RBRACE)
        else:
            then_branch = self.parse_statement()

        # 解析else分支
        else_branch = None
        if self.peek() and self.peek().type == TokenType.ELSE:
            self.advance()  # 跳过 'else'
            if self.peek().type == TokenType.IF:
                else_branch = self.parse_if_statement()  # 处理 else if
            elif self.peek().type == TokenType.LBRACE:
                self.advance()  # 跳过 '{'
                else_branch = self.parse_block()
                self.expect(TokenType.RBRACE)
            else:
                else_branch = self.parse_statement()

        return ASTNode(ASTNodeType.IF_STMT, children=[condition, then_branch, else_branch] if else_branch else [condition, then_branch])

    def parse_for_statement(self) -> ASTNode:
        """解析for语句"""
        self.advance()  # 跳过 'for'
        self.expect(TokenType.LPAREN)

        # 初始化部分
        initializer = None
        if self.peek().type != TokenType.SEMICOLON:
            if self.peek().type == TokenType.VAR:
                initializer = self.parse_var_declaration()
            else:
                initializer = self.parse_expression()
        self.expect(TokenType.SEMICOLON)

        # 条件部分
        condition = None
        if self.peek().type != TokenType.SEMICOLON:
            condition = self.parse_expression()
        self.expect(TokenType.SEMICOLON)

        # 增量部分
        increment = None
        if self.peek().type != TokenType.RPAREN:
            increment = self.parse_expression()
        self.expect(TokenType.RPAREN)

        # 循环体
        body = None
        if self.peek().type == TokenType.LBRACE:
            self.advance()  # 跳过 '{'
            body = self.parse_block()
            self.expect(TokenType.RBRACE)
        else:
            body = self.parse_statement()

        return ASTNode(ASTNodeType.FOR_STMT, children=[initializer, condition, increment, body])

    def parse_while_statement(self) -> ASTNode:
        """解析while语句"""
        self.advance()  # 跳过 'while'
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)

        # 循环体
        body = None
        if self.peek().type == TokenType.LBRACE:
            self.advance()  # 跳过 '{'
            body = self.parse_block()
            self.expect(TokenType.RBRACE)
        else:
            body = self.parse_statement()

        return ASTNode(ASTNodeType.WHILE_STMT, children=[condition, body])

    def parse_block(self) -> ASTNode:
        """解析代码块"""
        statements = []
        while not self.is_at_end() and self.peek().type != TokenType.RBRACE:
            statements.append(self.parse_statement())
            
            # 处理可选的分号
            if self.peek() and self.peek().type == TokenType.SEMICOLON:
                self.advance()

        return ASTNode(ASTNodeType.BLOCK, children=statements)
