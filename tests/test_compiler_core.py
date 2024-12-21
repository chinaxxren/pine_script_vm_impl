"""
测试编译器核心功能
"""

import unittest
from src.compiler.lexer import Lexer, TokenType
from src.compiler.parser import Parser, ASTNode, ASTNodeType
from src.compiler.compiler import Compiler
from src.vm.pine_vm import OpCode, Instruction

class TestCompilerCore(unittest.TestCase):
    """测试编译器核心功能"""

    def setUp(self):
        """测试前准备"""
        self.source = """
//@version=5
strategy("Test Strategy")

// 变量定义
var float myVar = 0.0
length = input(14, "Length")

// 指标计算
sma = ta.sma(close, length)
rsi = ta.rsi(close, length)

// 交易逻辑
if ta.crossover(sma, close)
    strategy.entry("Buy", strategy.long)
else if ta.crossunder(sma, close)
    strategy.close("Buy")
"""
        self.lexer = Lexer(self.source)
        self.tokens = list(self.lexer.tokenize())
        print("\nTokens:")
        for token in self.tokens:
            print(f"{token.type}: {token.value}")
        self.parser = Parser(self.tokens)
        self.ast = self.parser.parse()
        self.compiler = Compiler()

    def test_lexer_tokenization(self):
        """测试词法分析器的标记化功能"""
        # 测试基本标记
        tokens = list(Lexer('//@version=5').tokenize())
        self.assertEqual(len(tokens), 4)
        self.assertEqual(tokens[0].type, TokenType.COMMENT)
        self.assertEqual(tokens[1].type, TokenType.VERSION)
        self.assertEqual(tokens[2].type, TokenType.EQUALS)
        self.assertEqual(tokens[3].type, TokenType.NUMBER)

        # 测试标识符和关键字
        tokens = list(Lexer('strategy("Test")').tokenize())
        self.assertEqual(len(tokens), 4)
        self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[1].type, TokenType.LEFT_PAREN)
        self.assertEqual(tokens[2].type, TokenType.STRING)
        self.assertEqual(tokens[3].type, TokenType.RIGHT_PAREN)

        # 测试运算符
        tokens = list(Lexer('a = b + c * d').tokenize())
        self.assertEqual(len(tokens), 7)
        self.assertEqual(tokens[1].type, TokenType.EQUALS)
        self.assertEqual(tokens[3].type, TokenType.PLUS)
        self.assertEqual(tokens[5].type, TokenType.MULTIPLY)

    def test_parser_ast_generation(self):
        """测试语法分析器的AST生成功能"""
        # 测试基本表达式
        parser = Parser(list(Lexer('a = b + c').tokenize()))
        ast = parser.parse()
        self.assertEqual(ast.type, ASTNodeType.PROGRAM)
        self.assertEqual(len(ast.children), 1)
        assignment = ast.children[0]
        self.assertEqual(assignment.type, ASTNodeType.ASSIGNMENT)
        self.assertEqual(len(assignment.children), 2)
        
        # 测试if语句
        parser = Parser(list(Lexer('if a > b\n    c = d\nelse\n    e = f').tokenize()))
        ast = parser.parse()
        if_stmt = ast.children[0]
        self.assertEqual(if_stmt.type, ASTNodeType.IF_STMT)
        self.assertEqual(len(if_stmt.children), 3)  # 条件、then分支、else分支

        # 测试函数调用
        parser = Parser(list(Lexer('ta.sma(close, 14)').tokenize()))
        ast = parser.parse()
        func_call = ast.children[0]
        self.assertEqual(func_call.type, ASTNodeType.FUNCTION_CALL)
        self.assertEqual(len(func_call.children), 3)  # 函数名、参数1、参数2

    def test_compiler_instruction_generation(self):
        """测试编译器的指令生成功能"""
        # 测试常量加载
        compiler = Compiler()
        const_index = compiler.add_constant(42.0)
        compiler.emit(OpCode.LOAD_CONST, const_index)
        self.assertEqual(len(compiler.instructions), 1)
        self.assertEqual(compiler.instructions[0].opcode, OpCode.LOAD_CONST)
        self.assertEqual(compiler.instructions[0].operand, const_index)

        # 测试变量存储和加载
        compiler = Compiler()
        const_index = compiler.add_constant(42.0)
        compiler.emit(OpCode.LOAD_CONST, const_index)
        compiler.emit(OpCode.STORE_VAR, "x")
        compiler.emit(OpCode.LOAD_VAR, "x")
        self.assertEqual(len(compiler.instructions), 3)
        self.assertEqual(compiler.instructions[1].opcode, OpCode.STORE_VAR)
        self.assertEqual(compiler.instructions[2].opcode, OpCode.LOAD_VAR)

        # 测试跳转指令
        compiler = Compiler()
        jump_pos = compiler.emit(OpCode.JUMP_IF_FALSE, None)
        compiler.emit(OpCode.LOAD_CONST, compiler.add_constant(1.0))
        compiler.patch_jump(jump_pos, len(compiler.instructions))
        self.assertEqual(len(compiler.instructions), 2)
        self.assertEqual(compiler.instructions[0].opcode, OpCode.JUMP_IF_FALSE)
        self.assertEqual(compiler.instructions[0].operand, 2)

    def test_full_compilation_pipeline(self):
        """测试完整的编译流程"""
        # 简单的算术表达式
        source = "a = 1 + 2 * 3"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        instructions = compiler.compile(ast)
        
        # 验证生成的指令序列
        expected_opcodes = [
            OpCode.LOAD_CONST,  # 加载1
            OpCode.LOAD_CONST,  # 加载2
            OpCode.LOAD_CONST,  # 加载3
            OpCode.MULTIPLY,    # 2 * 3
            OpCode.ADD,         # 1 + (2 * 3)
            OpCode.STORE_VAR    # 存储到a
        ]
        self.assertEqual(len(instructions), len(expected_opcodes))
        for i, opcode in enumerate(expected_opcodes):
            self.assertEqual(instructions[i].opcode, opcode)

        # 测试if语句编译
        source = """
if close > open
    buy = 1
else
    buy = 0
"""
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        instructions = compiler.compile(ast)
        
        # 验证生成的指令包含正确的跳转
        self.assertTrue(any(i.opcode == OpCode.JUMP_IF_FALSE for i in instructions))
        self.assertTrue(any(i.opcode == OpCode.JUMP for i in instructions))

        # 测试函数调用编译
        source = "result = ta.sma(close, 14)"
        lexer = Lexer(source)
        tokens = list(lexer.tokenize())
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        instructions = compiler.compile(ast)
        
        # 验证生成的函数调用指令
        self.assertTrue(any(i.opcode == OpCode.CALL_FUNCTION for i in instructions))

if __name__ == '__main__':
    unittest.main()
