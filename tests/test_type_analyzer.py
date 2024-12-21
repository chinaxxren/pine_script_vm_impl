import unittest
from typing import List, Dict, Set, Optional, Tuple
from src.compiler.compiler import TypeAnalyzer, CompilerDiagnostics
from src.compiler.parser import Parser, ASTNode, ASTNodeType
from src.compiler.lexer import Lexer

class TestTypeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.type_analyzer = TypeAnalyzer()
        
    def parse_and_analyze(self, code: str) -> str:
        """解析代码并进行类型分析"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        return self.type_analyzer.analyze(ast)
        
    def test_basic_types(self):
        """测试基本类型分析"""
        # 测试数字类型
        node = ASTNode(ASTNodeType.NUMBER)
        node.value = 42
        self.assertEqual(self.type_analyzer.analyze(node), "int")
        
        node.value = 3.14
        self.assertEqual(self.type_analyzer.analyze(node), "float")
        
        # 测试字符串类型
        node = ASTNode(ASTNodeType.STRING)
        node.value = "hello"
        self.assertEqual(self.type_analyzer.analyze(node), "string")
        
        # 测试布尔类型
        node = ASTNode(ASTNodeType.IDENTIFIER)  # Pine Script中没有直接的布尔字面量
        node.value = "true"
        self.assertEqual(self.type_analyzer.analyze(node), "bool")
        
    def test_array_types(self):
        """测试数组类型分析"""
        # 创建数组字面量节点
        array_node = ASTNode(ASTNodeType.ARRAY_LITERAL)
        array_node.line = 1
        array_node.column = 1
        
        # 测试空数组
        self.assertEqual(self.type_analyzer.analyze(array_node), "array<any>")
        
        # 测试整数数组
        num1 = ASTNode(ASTNodeType.NUMBER)
        num1.value = 1
        num2 = ASTNode(ASTNodeType.NUMBER)
        num2.value = 2
        array_node.children = [num1, num2]
        self.assertEqual(self.type_analyzer.analyze(array_node), "array<int>")
        
        # 测试混合类型数组（应该报错）
        str_node = ASTNode(ASTNodeType.STRING)
        str_node.value = "hello"
        array_node.children.append(str_node)
        self.type_analyzer.analyze(array_node)
        self.assertTrue(any("Array elements must have compatible types" in err 
                          for err in self.type_analyzer.type_errors))
        
    def test_binary_operations(self):
        """测试二元运算符类型检查"""
        # 测试算术运算
        self.assertEqual(
            self.type_analyzer.check_operator_types("+", "int", "int"),
            "int"
        )
        self.assertEqual(
            self.type_analyzer.check_operator_types("+", "int", "float"),
            "float"
        )
        self.assertEqual(
            self.type_analyzer.check_operator_types("+", "string", "string"),
            "string"
        )
        
        # 测试比较运算
        self.assertEqual(
            self.type_analyzer.check_operator_types("<", "int", "int"),
            "bool"
        )
        self.assertEqual(
            self.type_analyzer.check_operator_types("==", "string", "string"),
            "bool"
        )
        
        # 测试逻辑运算
        self.assertEqual(
            self.type_analyzer.check_operator_types("&&", "bool", "bool"),
            "bool"
        )
        self.assertIsNone(
            self.type_analyzer.check_operator_types("&&", "int", "bool")
        )
        
    def test_function_types(self):
        """测试函数类型分析"""
        # 创建函数定义节点
        func_node = ASTNode(ASTNodeType.FUNCTION_DEFINITION)
        func_node.value = "test_func"
        func_node.params = ["x", "y"]
        func_node.line = 1
        func_node.column = 1
        
        # 创建函数体
        body = ASTNode(ASTNodeType.BLOCK)
        return_stmt = ASTNode(ASTNodeType.RETURN_STMT)
        num_node = ASTNode(ASTNodeType.NUMBER)
        num_node.value = 42
        return_stmt.children = [num_node]
        body.children = [return_stmt]
        func_node.body = body
        
        # 分析函数类型
        self.type_analyzer.analyze(func_node)
        func_type = self.type_analyzer.type_info["test_func"]
        self.assertTrue(func_type.startswith("function("))
        self.assertTrue(func_type.endswith("-> int"))
        
    def test_builtin_functions(self):
        """测试内置函数类型分析"""
        # 测试技术分析函数
        call_node = ASTNode(ASTNodeType.TA_FUNCTION)
        call_node.value = "sma"
        call_node.line = 1
        call_node.column = 1
        num_node = ASTNode(ASTNodeType.NUMBER)
        num_node.value = 14
        call_node.children = [num_node]
        
        self.assertEqual(self.type_analyzer.analyze(call_node), "float")
        
        # 测试交叉函数
        call_node.value = "cross"
        call_node.children = [num_node, num_node]  # 两个数值参数
        self.assertEqual(self.type_analyzer.analyze(call_node), "bool")
        
        # 测试绘图函数
        plot_node = ASTNode(ASTNodeType.PLOT_STATEMENT)
        plot_node.children = [num_node]
        self.assertEqual(self.type_analyzer.analyze(plot_node), "void")
        
    def test_type_compatibility(self):
        """测试类型兼容性检查"""
        self.assertTrue(
            self.type_analyzer.is_type_compatible("float", "int")
        )
        self.assertFalse(
            self.type_analyzer.is_type_compatible("int", "float")
        )
        self.assertTrue(
            self.type_analyzer.is_type_compatible("array<float>", "array<int>")
        )
        self.assertFalse(
            self.type_analyzer.is_type_compatible("array<int>", "array<string>")
        )
        
    def test_type_errors(self):
        """测试类型错误检测"""
        # 测试未定义变量
        node = ASTNode(ASTNodeType.IDENTIFIER)
        node.value = "undefined_var"
        node.line = 1
        node.column = 1
        self.type_analyzer.analyze(node)
        self.assertTrue(any("Undefined variable" in err 
                          for err in self.type_analyzer.type_errors))
        
        # 测试类型不匹配的赋值
        assign_node = ASTNode(ASTNodeType.ASSIGNMENT)
        assign_node.target = "x"
        assign_node.line = 1
        assign_node.column = 1
        str_node = ASTNode(ASTNodeType.STRING)
        str_node.value = "hello"
        assign_node.value = str_node
        
        # 先声明x为int类型
        self.type_analyzer.type_info["x"] = "int"
        self.type_analyzer.analyze(assign_node)
        self.assertTrue(any("Cannot assign value of type" in err 
                          for err in self.type_analyzer.type_errors))

if __name__ == '__main__':
    unittest.main()
