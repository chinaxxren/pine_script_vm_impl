import unittest
from src.compiler.compiler import CompilerDiagnostics

class TestCompilerDiagnostics(unittest.TestCase):
    def setUp(self):
        self.diagnostics = CompilerDiagnostics()
        
    def test_error_handling(self):
        """测试错误处理"""
        # 添加错误
        self.diagnostics.add_error("Division by zero", 10, 5)
        self.assertEqual(len(self.diagnostics.errors), 1)
        self.assertTrue("Division by zero" in self.diagnostics.errors[0])
        self.assertTrue("line 10, column 5" in self.diagnostics.errors[0])
        
        # 检查是否有错误
        self.assertTrue(self.diagnostics.has_errors())
        
    def test_warning_handling(self):
        """测试警告处理"""
        # 添加警告
        self.diagnostics.add_warning("Unused variable", 15, 3)
        self.assertEqual(len(self.diagnostics.warnings), 1)
        self.assertTrue("Unused variable" in self.diagnostics.warnings[0])
        self.assertTrue("line 15, column 3" in self.diagnostics.warnings[0])
        
        # 没有错误时不应报告有错误
        self.assertFalse(self.diagnostics.has_errors())
        
    def test_type_error_handling(self):
        """测试类型错误处理"""
        # 添加类型错误
        self.diagnostics.add_type_error("Type mismatch", 20, 8)
        self.assertEqual(len(self.diagnostics.type_errors), 1)
        self.assertTrue("Type mismatch" in self.diagnostics.type_errors[0])
        self.assertTrue("line 20, column 8" in self.diagnostics.type_errors[0])
        
        # 类型错误也应该被认为是错误
        self.assertTrue(self.diagnostics.has_errors())
        
    def test_optimization_info(self):
        """测试优化信息处理"""
        # 添加优化信息
        self.diagnostics.add_optimization_info("Constant folding applied")
        self.assertEqual(len(self.diagnostics.optimization_info), 1)
        self.assertTrue("Constant folding applied" in self.diagnostics.optimization_info[0])
        
    def test_info_handling(self):
        """测试一般信息处理"""
        # 添加一般信息
        self.diagnostics.add_info("Compilation started")
        self.assertEqual(len(self.diagnostics.info), 1)
        self.assertTrue("Compilation started" in self.diagnostics.info[0])
        
    def test_message_formatting(self):
        """测试消息格式化"""
        # 添加各种类型的消息
        self.diagnostics.add_error("Error 1", 1, 1)
        self.diagnostics.add_warning("Warning 1", 2, 2)
        self.diagnostics.add_type_error("Type Error 1", 3, 3)
        self.diagnostics.add_optimization_info("Optimization 1")
        self.diagnostics.add_info("Info 1")
        
        # 获取格式化的消息
        formatted = self.diagnostics.format_messages()
        
        # 验证格式
        self.assertTrue("Errors:" in formatted)
        self.assertTrue("Type Errors:" in formatted)
        self.assertTrue("Warnings:" in formatted)
        self.assertTrue("Optimization Info:" in formatted)
        self.assertTrue("Additional Info:" in formatted)
        
        # 验证消息内容
        self.assertTrue("Error 1" in formatted)
        self.assertTrue("Warning 1" in formatted)
        self.assertTrue("Type Error 1" in formatted)
        self.assertTrue("Optimization 1" in formatted)
        self.assertTrue("Info 1" in formatted)
        
    def test_multiple_messages(self):
        """测试多条消息处理"""
        # 添加多条错误
        self.diagnostics.add_error("Error 1", 1, 1)
        self.diagnostics.add_error("Error 2", 2, 2)
        
        # 添加多条警告
        self.diagnostics.add_warning("Warning 1", 3, 3)
        self.diagnostics.add_warning("Warning 2", 4, 4)
        
        # 验证消息数量
        self.assertEqual(len(self.diagnostics.errors), 2)
        self.assertEqual(len(self.diagnostics.warnings), 2)
        
        # 验证格式化输出包含所有消息
        formatted = self.diagnostics.format_messages()
        self.assertTrue("Error 1" in formatted)
        self.assertTrue("Error 2" in formatted)
        self.assertTrue("Warning 1" in formatted)
        self.assertTrue("Warning 2" in formatted)
        
    def test_empty_messages(self):
        """测试空消息处理"""
        # 没有添加任何消息时的格式化输出
        formatted = self.diagnostics.format_messages()
        self.assertEqual(formatted, "")
        
        # 没有错误时的状态检查
        self.assertFalse(self.diagnostics.has_errors())

if __name__ == '__main__':
    unittest.main()
