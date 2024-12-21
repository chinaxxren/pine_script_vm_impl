"""
编译器测试
"""

import unittest
from src.compiler.lexer import Lexer, TokenType
from src.compiler.parser import Parser, ASTNode
from src.compiler.compiler import Compiler

class TestLexer(unittest.TestCase):
    """词法分析器测试"""
    
    def setUp(self):
        self.lexer = Lexer()
        
    def test_basic_tokens(self):
        """测试基本 token"""
        code = "a = 1 + 2 * 3"
        tokens = self.lexer.tokenize(code)
        
        expected_types = [
            TokenType.IDENTIFIER,
            TokenType.ASSIGN,
            TokenType.NUMBER,
            TokenType.PLUS,
            TokenType.NUMBER,
            TokenType.MULTIPLY,
            TokenType.NUMBER
        ]
        
        self.assertEqual(len(tokens), len(expected_types))
        for token, expected_type in zip(tokens, expected_types):
            self.assertEqual(token.type, expected_type)
            
    def test_string_literals(self):
        """测试字符串字面量"""
        code = 'title = "Hello World"'
        tokens = self.lexer.tokenize(code)
        
        self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[1].type, TokenType.ASSIGN)
        self.assertEqual(tokens[2].type, TokenType.STRING)
        self.assertEqual(tokens[2].value, "Hello World")
        
    def test_keywords(self):
        """测试关键字"""
        code = "if close > open\n    strategy.entry('Long', strategy.long)\nelse\n    strategy.close('Long')"
        tokens = self.lexer.tokenize(code)
        
        keywords = ['if', 'else', 'strategy']
        for token in tokens:
            if token.value in keywords:
                self.assertEqual(token.type, TokenType.KEYWORD)
                
    def test_operators(self):
        """测试运算符"""
        code = "a = 1 + 2 - 3 * 4 / 5 % 6"
        tokens = self.lexer.tokenize(code)
        
        operators = ['+', '-', '*', '/', '%']
        operator_tokens = [t for t in tokens if t.value in operators]
        self.assertEqual(len(operator_tokens), len(operators))
        
    def test_comments(self):
        """测试注释"""
        code = "// This is a comment\na = 1 // Another comment"
        tokens = self.lexer.tokenize(code)
        
        # 注释应该被忽略
        self.assertEqual(len(tokens), 3)  # 只有 a, =, 1
        
class TestParser(unittest.TestCase):
    """语法分析器测试"""
    
    def setUp(self):
        self.parser = Parser()
        
    def test_basic_expression(self):
        """测试基本表达式"""
        code = "a = 1 + 2 * 3"
        ast = self.parser.parse(code)
        
        # 验证 AST 结构
        self.assertEqual(ast.type, "program")
        self.assertEqual(len(ast.body), 1)
        
        assign = ast.body[0]
        self.assertEqual(assign.type, "assignment")
        self.assertEqual(assign.left.type, "identifier")
        self.assertEqual(assign.left.value, "a")
        
        expr = assign.right
        self.assertEqual(expr.type, "binary")
        
    def test_if_statement(self):
        """测试 if 语句"""
        code = "if close > open\n    buy := true\nelse\n    buy := false"
        ast = self.parser.parse(code)
        
        # 验证 if 语句结构
        if_stmt = ast.body[0]
        self.assertEqual(if_stmt.type, "if")
        self.assertTrue(hasattr(if_stmt, "condition"))
        self.assertTrue(hasattr(if_stmt, "consequent"))
        self.assertTrue(hasattr(if_stmt, "alternate"))
        
    def test_function_call(self):
        """测试函数调用"""
        code = "sma = ta.sma(close, 14)"
        ast = self.parser.parse(code)
        
        # 验证函数调用结构
        assign = ast.body[0]
        call = assign.right
        self.assertEqual(call.type, "call")
        self.assertEqual(len(call.arguments), 2)
        
    def test_strategy_commands(self):
        """测试策略命令"""
        code = "strategy.entry('Long', strategy.long, when = crossover(fast, slow))"
        ast = self.parser.parse(code)
        
        # 验证策略命令结构
        cmd = ast.body[0]
        self.assertEqual(cmd.type, "call")
        self.assertEqual(cmd.callee.type, "member")
        
class TestCompiler(unittest.TestCase):
    """编译器测试"""
    
    def setUp(self):
        self.compiler = Compiler()
        
    def test_basic_compilation(self):
        """测试基本编译"""
        code = """
//@version=5
strategy("Test Strategy", overlay=true)

fastLength = input(10, "Fast Length")
slowLength = input(20, "Slow Length")

fast = ta.sma(close, fastLength)
slow = ta.sma(close, slowLength)

if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
else if ta.crossunder(fast, slow)
    strategy.close("Long")
"""
        # 编译代码
        result = self.compiler.compile(code)
        
        # 验证编译结果
        self.assertTrue(hasattr(result, "ast"))
        self.assertTrue(hasattr(result, "symbol_table"))
        self.assertTrue("fastLength" in result.symbol_table)
        self.assertTrue("slowLength" in result.symbol_table)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试语法错误
        code = "if close > open strategy.entry"  # 缺少换行和括号
        with self.assertRaises(Exception):
            self.compiler.compile(code)
            
        # 测试未定义变量
        code = "strategy.entry('Long', qty=position_size)"  # position_size 未定义
        with self.assertRaises(Exception):
            self.compiler.compile(code)
            
    def test_builtin_functions(self):
        """测试内置函数"""
        code = """
length = input(14)
rsi = ta.rsi(close, length)
upper = ta.highest(high, length)
lower = ta.lowest(low, length)
"""
        result = self.compiler.compile(code)
        
        # 验证内置函数是否正确解析
        self.assertTrue("ta.rsi" in str(result.ast))
        self.assertTrue("ta.highest" in str(result.ast))
        self.assertTrue("ta.lowest" in str(result.ast))
        
    def test_variable_scope(self):
        """测试变量作用域"""
        code = """
var float myVar = 0
if close > open
    myVar := close
else
    myVar := open
"""
        result = self.compiler.compile(code)
        
        # 验证变量声明和作用域
        self.assertTrue("myVar" in result.symbol_table)
        self.assertEqual(result.symbol_table["myVar"]["type"], "float")
        
if __name__ == '__main__':
    unittest.main()
