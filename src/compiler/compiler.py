from typing import List, Dict, Set, Any, Optional, Tuple
from ..vm.pine_vm import Instruction, OpCode
from .parser import ASTNode, ASTNodeType

class CompileError(Exception):
    """编译错误"""
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"{message} at line {line}, column {column}")

class Compiler:
    def __init__(self):
        self.instructions: List[Instruction] = []
        self.constants: List[Any] = []
        self.variables: Dict[str, int] = {}
        self.functions: Dict[str, int] = {}
        self.current_function: Optional[str] = None
        self.break_stack: List[int] = []
        self.continue_stack: List[int] = []
        self.scope_level: int = 0
        self.error_handlers: List[int] = []
        self.loop_depth: int = 0
        self.current_line: int = 0
        self.current_column: int = 0
        self.source_map: Dict[int, Tuple[int, int]] = {}  # 指令索引 -> (行号, 列号)
        self.optimization_level: int = 2  # 0: 无优化, 1: 基本优化, 2: 完全优化

    def compile(self, source: str, optimization_level: int = 2) -> List[Instruction]:
        """编译源代码"""
        self.optimization_level = optimization_level
        try:
            # 词法分析
            lexer = Lexer(source)
            tokens = list(lexer.tokenize())
            
            # 语法分析
            parser = Parser(tokens)
            ast = parser.parse()
            
            # 类型检查
            self.check_types(ast)
            
            # 代码生成
            self.generate_code(ast)
            
            # 代码优化
            if optimization_level > 0:
                self.optimize()
            
            # 验证生成的代码
            self.validate_code()
            
            return self.instructions
            
        except CompileError as e:
            raise e
        except Exception as e:
            raise CompileError(str(e), self.current_line, self.current_column)

    def check_types(self, node: ASTNode):
        """类型检查"""
        if node.type == ASTNodeType.BINARY_OP:
            # 检查操作数类型
            left_type = self.get_expression_type(node.children[0])
            right_type = self.get_expression_type(node.children[1])
            
            if not self.are_types_compatible(node.value, left_type, right_type):
                raise CompileError(
                    f"Type mismatch for operator '{node.value}': "
                    f"cannot operate on types {left_type} and {right_type}",
                    node.line, node.column
                )
                
        elif node.type == ASTNodeType.CALL:
            # 检查函数调用参数
            if node.value in self.functions:
                expected_params = self.get_function_params(node.value)
                if len(node.children) != len(expected_params):
                    raise CompileError(
                        f"Function '{node.value}' expects {len(expected_params)} "
                        f"arguments, but got {len(node.children)}",
                        node.line, node.column
                    )
                    
                for i, (param, arg) in enumerate(zip(expected_params, node.children)):
                    arg_type = self.get_expression_type(arg)
                    if not self.is_type_assignable(param, arg_type):
                        raise CompileError(
                            f"Type mismatch in argument {i+1} of function '{node.value}': "
                            f"expected {param}, got {arg_type}",
                            arg.line, arg.column
                        )
                        
        # 递归检查子节点
        for child in node.children:
            self.check_types(child)

    def get_expression_type(self, node: ASTNode) -> str:
        """获取表达式的类型"""
        if node.type == ASTNodeType.NUMBER:
            # 检查是否是整数
            if float(node.value).is_integer():
                return "int"
            return "float"
            
        elif node.type == ASTNodeType.STRING:
            return "string"
            
        elif node.type == ASTNodeType.BOOLEAN:
            return "bool"
            
        elif node.type == ASTNodeType.ARRAY_LITERAL:
            if node.children:
                element_type = self.get_expression_type(node.children[0])
                return f"array<{element_type}>"
            return "array<any>"
            
        elif node.type == ASTNodeType.IDENTIFIER:
            if node.value in self.variables:
                return self.get_variable_type(node.value)
            raise CompileError(
                f"Undefined variable '{node.value}'",
                node.line, node.column
            )
            
        elif node.type == ASTNodeType.BINARY_OP:
            # 根据操作符推断类型
            if node.value in ['<', '>', '<=', '>=', '==', '!=', '&&', '||']:
                return "bool"
            
            left_type = self.get_expression_type(node.children[0])
            right_type = self.get_expression_type(node.children[1])
            
            if left_type == right_type:
                return left_type
            
            if left_type in ['int', 'float'] and right_type in ['int', 'float']:
                return 'float'
                
            raise CompileError(
                f"Cannot determine result type for operator '{node.value}' "
                f"with operands of type {left_type} and {right_type}",
                node.line, node.column
            )
            
        elif node.type == ASTNodeType.CALL:
            if node.value in self.functions:
                return self.get_function_return_type(node.value)
            
            # 处理内置函数
            if node.value in ['sma', 'ema', 'rsi', 'macd']:
                return 'float'
            elif node.value in ['cross', 'crossover', 'crossunder']:
                return 'bool'
                
            raise CompileError(
                f"Unknown function '{node.value}'",
                node.line, node.column
            )
            
        raise CompileError(
            f"Cannot determine type for node {node.type}",
            node.line, node.column
        )

    def are_types_compatible(self, operator: str, left_type: str, right_type: str) -> bool:
        """检查类型是否兼容"""
        # 数值运算
        if operator in ['+', '-', '*', '/', '%', '**']:
            return (left_type in ['int', 'float'] and 
                   right_type in ['int', 'float'])
        
        # 比较运算
        elif operator in ['<', '>', '<=', '>=']:
            return (left_type in ['int', 'float'] and 
                   right_type in ['int', 'float']) or \
                   (left_type == right_type == 'string')
        
        # 相等性比较
        elif operator in ['==', '!=']:
            return left_type == right_type
        
        # 逻辑运算
        elif operator in ['&&', '||']:
            return left_type == right_type == 'bool'
        
        # 位运算
        elif operator in ['&', '|', '^', '<<', '>>']:
            return left_type == right_type == 'int'
        
        return False

    def validate_code(self):
        """验证生成的代码"""
        # 检查跳转目标是否有效
        for i, instr in enumerate(self.instructions):
            if instr.opcode in [OpCode.JMP, OpCode.JZ, OpCode.JNZ]:
                if not (0 <= instr.operand < len(self.instructions)):
                    raise CompileError(
                        f"Invalid jump target {instr.operand}",
                        self.source_map[i][0],
                        self.source_map[i][1]
                    )
        
        # 检查函数调用是否有效
        for i, instr in enumerate(self.instructions):
            if instr.opcode == OpCode.CALL:
                if isinstance(instr.operand, int):
                    if not (0 <= instr.operand < len(self.instructions)):
                        raise CompileError(
                            f"Invalid function call target {instr.operand}",
                            self.source_map[i][0],
                            self.source_map[i][1]
                        )
        
        # 检查栈平衡
        stack_depth = 0
        max_stack = 0
        for i, instr in enumerate(self.instructions):
            delta = self.get_stack_effect(instr)
            stack_depth += delta
            max_stack = max(max_stack, stack_depth)
            
            if stack_depth < 0:
                raise CompileError(
                    "Stack underflow",
                    self.source_map[i][0],
                    self.source_map[i][1]
                )

    def get_stack_effect(self, instr: Instruction) -> int:
        """计算指令对栈深度的影响"""
        effects = {
            OpCode.PUSH: 1,
            OpCode.POP: -1,
            OpCode.DUP: 1,
            OpCode.SWAP: 0,
            OpCode.LOAD: 1,
            OpCode.STORE: -1,
            OpCode.LOAD_CONST: 1,
            OpCode.ADD: -1,
            OpCode.SUB: -1,
            OpCode.MUL: -1,
            OpCode.DIV: -1,
            OpCode.MOD: -1,
            OpCode.POW: -1,
            OpCode.NEG: 0,
            OpCode.AND: -1,
            OpCode.OR: -1,
            OpCode.XOR: -1,
            OpCode.NOT: 0,
            OpCode.SHL: -1,
            OpCode.SHR: -1,
            OpCode.EQ: -1,
            OpCode.NE: -1,
            OpCode.LT: -1,
            OpCode.LE: -1,
            OpCode.GT: -1,
            OpCode.GE: -1,
            OpCode.LAND: -1,
            OpCode.LOR: -1,
            OpCode.LNOT: 0,
            OpCode.JMP: 0,
            OpCode.JZ: -1,
            OpCode.JNZ: -1,
            OpCode.CALL: 0,  # 具体值取决于函数定义
            OpCode.RET: -1,
            OpCode.NEW_ARRAY: 0,  # 取决于数组大小
            OpCode.ARRAY_GET: -1,
            OpCode.ARRAY_SET: -2,
            OpCode.ARRAY_LEN: 0,
        }
        return effects.get(instr.opcode, 0)

    def optimize(self):
        """优化生成的代码"""
        if self.optimization_level >= 1:
            # 基本优化
            self.fold_constants()
            self.eliminate_dead_code()
            
        if self.optimization_level >= 2:
            # 高级优化
            self.peephole_optimize()
            self.optimize_jumps()
            self.eliminate_redundant_loads()
            self.optimize_array_operations()

    def optimize_jumps(self):
        """优化跳转指令"""
        # 跳转到跳转的优化
        changed = True
        while changed:
            changed = False
            for i, instr in enumerate(self.instructions):
                if instr.opcode in [OpCode.JMP, OpCode.JZ, OpCode.JNZ]:
                    target = instr.operand
                    while (target < len(self.instructions) and 
                           self.instructions[target].opcode == OpCode.JMP):
                        target = self.instructions[target].operand
                    if target != instr.operand:
                        self.instructions[i].operand = target
                        changed = True

    def eliminate_redundant_loads(self):
        """消除冗余的加载指令"""
        i = 0
        while i < len(self.instructions) - 1:
            if (self.instructions[i].opcode == OpCode.LOAD and
                self.instructions[i + 1].opcode == OpCode.LOAD and
                self.instructions[i].operand == self.instructions[i + 1].operand):
                # 删除第二个加载
                del self.instructions[i + 1]
                continue
            i += 1

    def optimize_array_operations(self):
        """优化数组操作"""
        i = 0
        while i < len(self.instructions) - 2:
            # 优化 array.get 后立即 array.set
            if (self.instructions[i].opcode == OpCode.ARRAY_GET and
                self.instructions[i + 1].opcode == OpCode.ARRAY_SET and
                self.instructions[i].operand == self.instructions[i + 1].operand):
                del self.instructions[i:i + 2]
                continue
                
            # 优化连续的数组长度检查
            if (self.instructions[i].opcode == OpCode.ARRAY_LEN and
                self.instructions[i + 1].opcode == OpCode.ARRAY_LEN and
                self.instructions[i].operand == self.instructions[i + 1].operand):
                del self.instructions[i + 1]
                continue
                
            i += 1

    def emit(self, opcode: OpCode, operand: Any = None):
        """发出指令"""
        instruction = Instruction(opcode, operand, self.current_line, self.current_column)
        self.instructions.append(instruction)
        self.source_map[len(self.instructions) - 1] = (self.current_line, self.current_column)

    def add_constant(self, value) -> int:
        """添加常量到常量池"""
        key = str(value)
        if key not in self.constants:
            self.constants.append(value)
        return self.constants.index(value)

    def add_name(self, name: str) -> int:
        """添加名称到名称池"""
        if name not in self.variables:
            self.variables[name] = len(self.variables)
        return self.variables[name]

    def patch_jump(self, position: int, target: int) -> None:
        """修复跳转指令的目标位置"""
        self.instructions[position].operand = target

    def compile_node(self, node: ASTNode) -> None:
        """编译单个AST节点"""
        if node.type == ASTNodeType.PROGRAM:
            for child in node.children:
                self.compile_node(child)

        elif node.type == ASTNodeType.VERSION_DECL:
            pass  # 版本声明不生成字节码

        elif node.type == ASTNodeType.INDICATOR_DECL:
            const_index = self.add_constant(node.value)
            self.emit(OpCode.LOAD_CONST, const_index)

        elif node.type == ASTNodeType.PLOT_STMT:
            # 编译plot语句的表达式
            self.compile_node(node.children[0])

            # 解析绘图选项
            options = {}
            if len(node.children) > 1:
                options = self.compile_plot_options(node.children[1])

            # 发出绘图指令
            self.emit(OpCode.PLOT, options)

        elif node.type == ASTNodeType.PLOT_CANDLE:
            # 编译K线数据
            for child in node.children[:4]:  # open, high, low, close
                self.compile_node(child)

            # 解析绘图选项
            options = {}
            if len(node.children) > 4:
                options = self.compile_plot_options(node.children[4])

            # 发出K线绘图指令
            self.emit(OpCode.PLOT_CANDLE, options)

        elif node.type == ASTNodeType.PLOT_SHAPE:
            # 编译位置和值
            self.compile_node(node.children[0])  # 位置
            self.compile_node(node.children[1])  # 值

            # 解析绘图选项
            options = {}
            if len(node.children) > 2:
                options = self.compile_plot_options(node.children[2])

            # 发出形状绘图指令
            self.emit(OpCode.PLOT_SHAPE, options)

        elif node.type == ASTNodeType.PLOT_ARROW:
            # 编译位置和方向
            self.compile_node(node.children[0])  # 位置
            self.compile_node(node.children[1])  # 方向

            # 解析绘图选项
            options = {}
            if len(node.children) > 2:
                options = self.compile_plot_options(node.children[2])

            # 发出箭头绘图指令
            self.emit(OpCode.PLOT_ARROW, options)

        elif node.type == ASTNodeType.PLOT_BGCOLOR:
            # 编译颜色值
            self.compile_node(node.children[0])

            # 解析绘图选项
            options = {}
            if len(node.children) > 1:
                options = self.compile_plot_options(node.children[1])

            # 发出背景色绘图指令
            self.emit(OpCode.PLOT_BGCOLOR, options)

        elif node.type == ASTNodeType.IF_STMT:
            # 编译条件表达式
            self.compile_node(node.children[0])

            # 发出条件跳转指令
            jump_false_pos = self.emit(OpCode.JUMP_IF_FALSE, None)

            # 编译then分支
            self.compile_node(node.children[1])

            if len(node.children) > 2:  # 有else分支
                jump_end_pos = self.emit(OpCode.JUMP, None)

                # 修复条件跳转位置
                self.patch_jump(jump_false_pos, len(self.instructions))

                # 编译else分支
                self.compile_node(node.children[2])

                # 修复无条件跳转位置
                self.patch_jump(jump_end_pos, len(self.instructions))
            else:
                # 修复条件跳转位置
                self.patch_jump(jump_false_pos, len(self.instructions))

        elif node.type == ASTNodeType.WHILE_STMT:
            loop_start = len(self.instructions)
            self.continue_targets.append(loop_start)

            # 编译条件
            self.compile_node(node.children[0])

            # 条件跳转
            jump_false_pos = self.emit(OpCode.JUMP_IF_FALSE, None)

            # 为break语句准备
            self.break_targets.append(None)

            # 编译循环体
            self.compile_node(node.children[1])

            # 跳回循环开始
            self.emit(OpCode.JUMP, loop_start)

            # 修复break目标
            loop_end = len(self.instructions)
            self.patch_jump(jump_false_pos, loop_end)

            # 修复所有break跳转
            break_pos = self.break_targets.pop()
            if break_pos is not None:
                self.patch_jump(break_pos, loop_end)

            self.continue_targets.pop()

        elif node.type == ASTNodeType.FOR_STMT:
            # 编译初始值
            self.compile_node(node.children[0])
            loop_var_pos = self.emit(OpCode.STORE_VAR, "_loop_var")

            loop_start = len(self.instructions)
            self.continue_targets.append(loop_start)

            # 加载循环变量
            self.emit(OpCode.LOAD_VAR, "_loop_var")

            # 编译结束值
            self.compile_node(node.children[1])

            # 比较
            self.emit(OpCode.COMPARISON, "<=")

            # 条件跳转
            jump_false_pos = self.emit(OpCode.JUMP_IF_FALSE, None)

            # 为break语句准备
            self.break_targets.append(None)

            # 编译循环体
            self.compile_node(node.children[2])

            # 增加循环变量
            self.emit(OpCode.LOAD_VAR, "_loop_var")
            self.emit(OpCode.LOAD_CONST, self.add_constant(1))
            self.emit(OpCode.ADD)
            self.emit(OpCode.STORE_VAR, "_loop_var")

            # 跳回循环开始
            self.emit(OpCode.JUMP, loop_start)

            # 修复break目标
            loop_end = len(self.instructions)
            self.patch_jump(jump_false_pos, loop_end)

            # 修复所有break跳转
            break_pos = self.break_targets.pop()
            if break_pos is not None:
                self.patch_jump(break_pos, loop_end)

            self.continue_targets.pop()

        elif node.type == ASTNodeType.BREAK_STMT:
            if not self.break_targets:
                raise SyntaxError("Break statement outside loop")
            pos = self.emit(OpCode.JUMP, None)
            self.break_targets[-1] = pos

        elif node.type == ASTNodeType.CONTINUE_STMT:
            if not self.continue_targets:
                raise SyntaxError("Continue statement outside loop")
            self.emit(OpCode.JUMP, self.continue_targets[-1])

        elif node.type == ASTNodeType.BLOCK:
            for child in node.children:
                self.compile_node(child)

        elif node.type == ASTNodeType.BINARY_OP:
            # 编译左操作数
            self.compile_node(node.children[0])
            # 编译右操作数
            self.compile_node(node.children[1])
            # 发出相应的操作码
            self.emit(self.get_binary_opcode(node.value))

        elif node.type == ASTNodeType.COMPARISON:
            self.compile_node(node.children[0])
            self.compile_node(node.children[1])
            self.emit(OpCode.COMPARISON, node.value)

        elif node.type == ASTNodeType.LOGICAL_OP:
            if node.value == 'and':
                # 短路与
                self.compile_node(node.children[0])
                jump_false_pos = self.emit(OpCode.JUMP_IF_FALSE_KEEP, None)
                self.compile_node(node.children[1])
                self.patch_jump(jump_false_pos, len(self.instructions))
            elif node.value == 'or':
                # 短路或
                self.compile_node(node.children[0])
                jump_true_pos = self.emit(OpCode.JUMP_IF_TRUE_KEEP, None)
                self.compile_node(node.children[1])
                self.patch_jump(jump_true_pos, len(self.instructions))

        elif node.type == ASTNodeType.NUMBER:
            const_index = self.add_constant(node.value)
            self.emit(OpCode.LOAD_CONST, const_index)

        elif node.type == ASTNodeType.IDENTIFIER:
            name_index = self.add_name(node.value)
            self.emit(OpCode.LOAD_NAME, name_index)

        elif node.type == ASTNodeType.ASSIGNMENT:
            # 编译赋值表达式
            self.compile_node(node.children[0])
            # 发出存储指令
            name_index = self.add_name(node.value)
            self.emit(OpCode.STORE_NAME, name_index)

        elif node.type == ASTNodeType.INDICATOR_CALL:
            # 编译指标参数
            for param in node.children[1:]:
                self.compile_node(param)

            # 获取指标名称和参数个数
            indicator_name = node.children[0].value
            param_count = len(node.children) - 1

            # 发出指标计算指令
            self.emit(OpCode.CALL_INDICATOR, (indicator_name, param_count))

        elif node.type == ASTNodeType.INDICATOR_PARAM:
            # 编译参数值
            self.compile_node(node.children[0])

            # 发出参数名称
            self.emit(OpCode.LOAD_CONST, node.value)

        elif node.type == ASTNodeType.STRATEGY_ENTRY:
            # 编译开仓参数
            self.compile_node(node.children[0])  # 数量
            self.compile_node(node.children[1])  # 价格
            self.compile_node(node.children[2])  # 方向

            # 发出开仓指令
            self.emit(OpCode.STRATEGY_ENTRY)

        elif node.type == ASTNodeType.STRATEGY_EXIT:
            # 编译平仓参数
            self.compile_node(node.children[0])  # 价格

            # 发出平仓指令
            self.emit(OpCode.STRATEGY_EXIT)

        elif node.type == ASTNodeType.STRATEGY_CLOSE:
            # 发出清仓指令
            self.emit(OpCode.STRATEGY_CLOSE)

        else:
            raise ValueError(f"Unknown node type: {node.type}")

    def get_binary_opcode(self, op: str) -> OpCode:
        """获取二元操作的操作码"""
        op_map = {
            '+': OpCode.ADD,
            '-': OpCode.SUB,
            '*': OpCode.MUL,
            '/': OpCode.DIV
        }
        return op_map[op]

    def compile_plot_options(self, options_node: ASTNode) -> dict:
        """编译绘图选项"""
        options = {}

        # 遍历选项节点的所有子节点
        for child in options_node.children:
            if child.type == ASTNodeType.PLOT_OPTION:
                key = child.value
                value = self.compile_plot_option_value(child.children[0])
                options[key] = value

        return options

    def compile_plot_option_value(self, value_node: ASTNode) -> Any:
        """编译绘图选项的值"""
        if value_node.type == ASTNodeType.STRING:
            return value_node.value
        elif value_node.type == ASTNodeType.NUMBER:
            return float(value_node.value)
        elif value_node.type == ASTNodeType.BOOLEAN:
            return value_node.value.lower() == 'true'
        elif value_node.type == ASTNodeType.COLOR:
            return value_node.value
        else:
            raise ValueError(f"Unsupported plot option value type: {value_node.type}")

    def fold_constants(self):
        """常量折叠优化"""
        i = 0
        while i < len(self.instructions) - 1:
            if (self.instructions[i].opcode == OpCode.LOAD_CONST and
                self.instructions[i + 1].opcode == OpCode.LOAD_CONST):
                # 获取两个常量
                const1 = self.constants[self.instructions[i].operand]
                const2 = self.constants[self.instructions[i + 1].operand]
                
                if i + 2 < len(self.instructions):
                    # 尝试计算结果
                    result = None
                    if self.instructions[i + 2].opcode == OpCode.ADD:
                        result = const1 + const2
                    elif self.instructions[i + 2].opcode == OpCode.SUB:
                        result = const1 - const2
                    elif self.instructions[i + 2].opcode == OpCode.MUL:
                        result = const1 * const2
                    elif self.instructions[i + 2].opcode == OpCode.DIV:
                        if const2 != 0:
                            result = const1 / const2
                            
                    if result is not None:
                        # 用一个LOAD_CONST指令替换这三个指令
                        const_index = self.add_constant(result)
                        self.instructions[i] = Instruction(OpCode.LOAD_CONST, const_index)
                        del self.instructions[i + 1:i + 3]
                        continue
            i += 1
    
    def eliminate_dead_code(self):
        """删除无用代码优化"""
        # 标记可达代码
        reachable = [False] * len(self.instructions)
        stack = [0]  # 从第一条指令开始
        
        while stack:
            i = stack.pop()
            if i >= len(self.instructions) or reachable[i]:
                continue
                
            reachable[i] = True
            instr = self.instructions[i]
            
            if instr.opcode in (OpCode.JMP, OpCode.CALL):
                stack.append(instr.operand)
                if instr.opcode == OpCode.JMP:
                    continue
                    
            elif instr.opcode in (OpCode.JZ, OpCode.JNZ):
                stack.append(instr.operand)
                stack.append(i + 1)
                continue
                
            stack.append(i + 1)
            
        # 删除不可达代码
        self.instructions = [instr for i, instr in enumerate(self.instructions) if reachable[i]]

    def peephole_optimize(self):
        """窥孔优化"""
        i = 0
        while i < len(self.instructions) - 1:
            # 优化 PUSH + POP
            if (self.instructions[i].opcode == OpCode.PUSH and
                self.instructions[i + 1].opcode == OpCode.POP):
                del self.instructions[i:i + 2]
                continue
                
            # 优化 JMP + JMP
            if (self.instructions[i].opcode == OpCode.JMP and
                i + 1 < len(self.instructions) and
                self.instructions[i + 1].opcode == OpCode.JMP):
                self.instructions[i].operand = self.instructions[i + 1].operand
                del self.instructions[i + 1]
                continue
                
            # 优化 NOT + NOT
            if (self.instructions[i].opcode == OpCode.NOT and
                self.instructions[i + 1].opcode == OpCode.NOT):
                del self.instructions[i:i + 2]
                continue
                
            i += 1

    def generate_code(self, node: ASTNode):
        """生成代码"""
        self.current_line = node.line
        self.current_column = node.column
        
        if node.type == ASTNodeType.PROGRAM:
            for child in node.children:
                self.generate_code(child)
                
        elif node.type == ASTNodeType.FUNCTION_DEF:
            self.generate_function_def(node)
            
        elif node.type == ASTNodeType.RETURN_STMT:
            self.generate_return_stmt(node)
            
        elif node.type == ASTNodeType.IF_STMT:
            self.generate_if_stmt(node)
            
        elif node.type == ASTNodeType.WHILE_STMT:
            self.generate_while_stmt(node)
            
        elif node.type == ASTNodeType.FOR_STMT:
            self.generate_for_stmt(node)
            
        elif node.type == ASTNodeType.BREAK_STMT:
            self.generate_break_stmt(node)
            
        elif node.type == ASTNodeType.CONTINUE_STMT:
            self.generate_continue_stmt(node)
            
        elif node.type == ASTNodeType.ASSIGNMENT:
            self.generate_assignment(node)
            
        elif node.type == ASTNodeType.CALL:
            self.generate_call(node)
            
        elif node.type == ASTNodeType.BINARY_OP:
            self.generate_binary_op(node)
            
        elif node.type == ASTNodeType.UNARY_OP:
            self.generate_unary_op(node)
            
        elif node.type == ASTNodeType.ARRAY_LITERAL:
            self.generate_array_literal(node)
            
        elif node.type == ASTNodeType.ARRAY_ACCESS:
            self.generate_array_access(node)
            
        elif node.type == ASTNodeType.PLOT_STMT:
            self.generate_plot_stmt(node)
            
        elif node.type == ASTNodeType.STRATEGY_STMT:
            self.generate_strategy_stmt(node)
            
        elif node.type == ASTNodeType.NUMBER:
            self.emit(OpCode.LOAD_CONST, self.add_constant(float(node.value)))
            
        elif node.type == ASTNodeType.STRING:
            self.emit(OpCode.LOAD_CONST, self.add_constant(node.value))
            
        elif node.type == ASTNodeType.BOOLEAN:
            self.emit(OpCode.LOAD_CONST, self.add_constant(node.value == 'true'))
            
        elif node.type == ASTNodeType.IDENTIFIER:
            self.emit(OpCode.LOAD, self.get_variable(node.value))
            
        else:
            raise CompileError(f"Unknown node type: {node.type}", node.line, node.column)

    def generate_function_def(self, node: ASTNode):
        """生成函数定义代码"""
        # 保存当前函数上下文
        old_function = self.current_function
        self.current_function = node.value
        
        # 记录函数开始位置
        function_start = len(self.instructions)
        self.functions[node.value] = function_start
        
        # 生成参数加载代码
        for param in node.params:
            self.variables[param] = len(self.variables)
            self.emit(OpCode.STORE, self.variables[param])
        
        # 生成函数体代码
        self.generate_code(node.body)
        
        # 如果没有显式的return语句，添加一个
        if not self.instructions or self.instructions[-1].opcode != OpCode.RET:
            self.emit(OpCode.LOAD_CONST, self.add_constant(None))
            self.emit(OpCode.RET)
        
        # 恢复函数上下文
        self.current_function = old_function

    def generate_return_stmt(self, node: ASTNode):
        """生成return语句代码"""
        if node.children:
            self.generate_code(node.children[0])
        else:
            self.emit(OpCode.LOAD_CONST, self.add_constant(None))
        self.emit(OpCode.RET)

    def generate_if_stmt(self, node: ASTNode):
        """生成if语句代码"""
        # 生成条件表达式代码
        self.generate_code(node.children[0])
        
        # 发出条件跳转指令
        jz_index = len(self.instructions)
        self.emit(OpCode.JZ, 0)  # 占位，后面回填
        
        # 生成then分支代码
        self.generate_code(node.children[1])
        
        if len(node.children) > 2:  # 有else分支
            # 发出无条件跳转指令
            jmp_index = len(self.instructions)
            self.emit(OpCode.JMP, 0)  # 占位，后面回填
            
            # 回填条件跳转的目标地址
            self.instructions[jz_index].operand = len(self.instructions)
            
            # 生成else分支代码
            self.generate_code(node.children[2])
            
            # 回填无条件跳转的目标地址
            self.instructions[jmp_index].operand = len(self.instructions)
        else:
            # 回填条件跳转的目标地址
            self.instructions[jz_index].operand = len(self.instructions)

    def generate_while_stmt(self, node: ASTNode):
        """生成while语句代码"""
        self.loop_depth += 1
        
        # 记录循环开始位置
        loop_start = len(self.instructions)
        
        # 生成条件表达式代码
        self.generate_code(node.children[0])
        
        # 发出条件跳转指令
        jz_index = len(self.instructions)
        self.emit(OpCode.JZ, 0)  # 占位，后面回填
        
        # 保存break和continue目标
        self.break_stack.append(0)  # 占位，后面回填
        self.continue_stack.append(loop_start)
        
        # 生成循环体代码
        self.generate_code(node.children[1])
        
        # 生成跳回循环开始的指令
        self.emit(OpCode.JMP, loop_start)
        
        # 回填条件跳转和break的目标地址
        end_index = len(self.instructions)
        self.instructions[jz_index].operand = end_index
        self.break_stack[-1] = end_index
        
        # 恢复break和continue目标
        self.break_stack.pop()
        self.continue_stack.pop()
        
        self.loop_depth -= 1

    def generate_for_stmt(self, node: ASTNode):
        """生成for语句代码"""
        self.loop_depth += 1
        
        # 生成初始化代码
        if node.init:
            self.generate_code(node.init)
        
        # 记录循环开始位置
        loop_start = len(self.instructions)
        
        # 生成条件表达式代码
        if node.condition:
            self.generate_code(node.condition)
        else:
            self.emit(OpCode.LOAD_CONST, self.add_constant(True))
        
        # 发出条件跳转指令
        jz_index = len(self.instructions)
        self.emit(OpCode.JZ, 0)  # 占位，后面回填
        
        # 保存break和continue目标
        self.break_stack.append(0)  # 占位，后面回填
        self.continue_stack.append(0)  # 占位，后面回填
        
        # 生成循环体代码
        self.generate_code(node.body)
        
        # 记录continue目标位置
        continue_target = len(self.instructions)
        self.continue_stack[-1] = continue_target
        
        # 生成更新代码
        if node.update:
            self.generate_code(node.update)
        
        # 生成跳回循环开始的指令
        self.emit(OpCode.JMP, loop_start)
        
        # 回填条件跳转和break的目标地址
        end_index = len(self.instructions)
        self.instructions[jz_index].operand = end_index
        self.break_stack[-1] = end_index
        
        # 恢复break和continue目标
        self.break_stack.pop()
        self.continue_stack.pop()
        
        self.loop_depth -= 1

    def generate_break_stmt(self, node: ASTNode):
        """生成break语句代码"""
        if not self.break_stack:
            raise CompileError("Break statement outside loop", node.line, node.column)
        self.emit(OpCode.JMP, self.break_stack[-1])

    def generate_continue_stmt(self, node: ASTNode):
        """生成continue语句代码"""
        if not self.continue_stack:
            raise CompileError("Continue statement outside loop", node.line, node.column)
        self.emit(OpCode.JMP, self.continue_stack[-1])

    def generate_assignment(self, node: ASTNode):
        """生成赋值语句代码"""
        if node.type == ASTNodeType.ARRAY_ACCESS:
            # 数组元素赋值
            self.generate_code(node.children[0])  # 数组
            self.generate_code(node.children[1])  # 索引
            self.generate_code(node.value)        # 值
            self.emit(OpCode.ARRAY_SET)
        else:
            # 普通变量赋值
            self.generate_code(node.value)
            self.emit(OpCode.STORE, self.get_variable(node.target))

    def generate_binary_op(self, node: ASTNode):
        """生成二元运算代码"""
        self.generate_code(node.children[0])
        self.generate_code(node.children[1])
        
        opcodes = {
            '+': OpCode.ADD,
            '-': OpCode.SUB,
            '*': OpCode.MUL,
            '/': OpCode.DIV,
            '%': OpCode.MOD,
            '**': OpCode.POW,
            '&': OpCode.AND,
            '|': OpCode.OR,
            '^': OpCode.XOR,
            '<<': OpCode.SHL,
            '>>': OpCode.SHR,
            '==': OpCode.EQ,
            '!=': OpCode.NE,
            '<': OpCode.LT,
            '<=': OpCode.LE,
            '>': OpCode.GT,
            '>=': OpCode.GE,
            '&&': OpCode.LAND,
            '||': OpCode.LOR,
        }
        
        if node.value not in opcodes:
            raise CompileError(f"Unknown operator: {node.value}", node.line, node.column)
            
        self.emit(opcodes[node.value])

    def generate_unary_op(self, node: ASTNode):
        """生成一元运算代码"""
        self.generate_code(node.children[0])
        
        opcodes = {
            '-': OpCode.NEG,
            '!': OpCode.LNOT,
            '~': OpCode.NOT,
        }
        
        if node.value not in opcodes:
            raise CompileError(f"Unknown operator: {node.value}", node.line, node.column)
            
        self.emit(opcodes[node.value])

    def generate_array_literal(self, node: ASTNode):
        """生成数组字面量代码"""
        # 生成数组长度
        self.emit(OpCode.LOAD_CONST, self.add_constant(len(node.children)))
        
        # 生成数组元素
        for child in node.children:
            self.generate_code(child)
            
        # 创建数组
        self.emit(OpCode.NEW_ARRAY)

    def generate_array_access(self, node: ASTNode):
        """生成数组访问代码"""
        # 生成数组对象代码
        self.generate_code(node.children[0])
        
        # 生成索引代码
        self.generate_code(node.children[1])
        
        # 生成数组访问指令
        self.emit(OpCode.ARRAY_GET)

    def generate_plot_stmt(self, node: ASTNode):
        """生成绘图语句代码"""
        # 生成系列值
        self.generate_code(node.series)
        
        # 生成绘图选项
        for option in node.options:
            self.generate_code(option.value)
            
        # 发出绘图指令
        plot_opcodes = {
            'plot': OpCode.PLOT,
            'plotarrow': OpCode.PLOT_ARROW,
            'plotshape': OpCode.PLOT_SHAPE,
            'plotchar': OpCode.PLOT_CHAR,
            'plotbar': OpCode.PLOT_BAR,
            'plotcandle': OpCode.PLOT_CANDLE,
            'plothistogram': OpCode.PLOT_HISTOGRAM,
        }
        
        if node.plot_type not in plot_opcodes:
            raise CompileError(f"Unknown plot type: {node.plot_type}", node.line, node.column)
            
        self.emit(plot_opcodes[node.plot_type])

    def generate_strategy_stmt(self, node: ASTNode):
        """生成策略语句代码"""
        # 生成条件代码
        if node.condition:
            self.generate_code(node.condition)
        else:
            self.emit(OpCode.LOAD_CONST, self.add_constant(True))
        
        # 生成数量代码
        if node.quantity:
            self.generate_code(node.quantity)
        else:
            self.emit(OpCode.LOAD_CONST, self.add_constant(1.0))
        
        # 生成价格代码
        if node.price:
            self.generate_code(node.price)
        else:
            self.emit(OpCode.LOAD_CONST, self.add_constant(None))
        
        # 生成止损价格代码
        if node.stop_loss:
            self.generate_code(node.stop_loss)
        else:
            self.emit(OpCode.LOAD_CONST, self.add_constant(None))
        
        # 生成止盈价格代码
        if node.take_profit:
            self.generate_code(node.take_profit)
        else:
            self.emit(OpCode.LOAD_CONST, self.add_constant(None))
        
        # 发出策略指令
        strategy_opcodes = {
            'strategy.entry': OpCode.STRATEGY_ENTRY,
            'strategy.order': OpCode.STRATEGY_ORDER,
            'strategy.exit': OpCode.STRATEGY_EXIT,
            'strategy.close': OpCode.STRATEGY_CLOSE,
            'strategy.cancel': OpCode.STRATEGY_CANCEL,
            'strategy.cancel_all': OpCode.STRATEGY_CANCEL_ALL,
        }
        
        if node.strategy_type not in strategy_opcodes:
            raise CompileError(f"Unknown strategy type: {node.strategy_type}", 
                             node.line, node.column)
            
        self.emit(strategy_opcodes[node.strategy_type])

class SymbolInfo:
    """符号信息"""
    def __init__(self, name: str, symbol_type: str, is_mutable: bool = True):
        self.name = name
        self.type = symbol_type
        self.is_mutable = is_mutable
        self.is_used = False
        self.is_initialized = False
        self.references = 0
        self.assignments = 0
        self.scope_level = 0
        self.line = 0
        self.column = 0

class ScopeAnalyzer:
    """作用域分析器"""
    def __init__(self):
        self.scopes: List[Dict[str, SymbolInfo]] = [{}]  # 作用域栈
        self.current_scope = 0
        self.warnings: List[str] = []
        
    def enter_scope(self):
        """进入新的作用域"""
        self.scopes.append({})
        self.current_scope += 1
        
    def exit_scope(self):
        """退出当前作用域"""
        # 检查未使用的变量
        for name, info in self.scopes[-1].items():
            if not info.is_used:
                self.warnings.append(
                    f"Warning: Variable '{name}' is never used "
                    f"at line {info.line}, column {info.column}"
                )
        self.scopes.pop()
        self.current_scope -= 1
        
    def declare(self, name: str, symbol_type: str, is_mutable: bool = True,
                line: int = 0, column: int = 0):
        """声明一个符号"""
        if name in self.scopes[-1]:
            raise CompileError(
                f"Symbol '{name}' is already declared in current scope",
                line, column
            )
        
        info = SymbolInfo(name, symbol_type, is_mutable)
        info.scope_level = self.current_scope
        info.line = line
        info.column = column
        self.scopes[-1][name] = info
        
    def lookup(self, name: str) -> Optional[SymbolInfo]:
        """查找符号"""
        # 从内层作用域向外层查找
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
        
    def reference(self, name: str, line: int, column: int):
        """记录符号引用"""
        info = self.lookup(name)
        if info:
            info.is_used = True
            info.references += 1
        else:
            raise CompileError(
                f"Reference to undefined symbol '{name}'",
                line, column
            )
            
    def assign(self, name: str, line: int, column: int):
        """记录符号赋值"""
        info = self.lookup(name)
        if info:
            if not info.is_mutable:
                raise CompileError(
                    f"Cannot assign to immutable symbol '{name}'",
                    line, column
                )
            info.is_initialized = True
            info.assignments += 1
        else:
            raise CompileError(
                f"Assignment to undefined symbol '{name}'",
                line, column
            )

class DataFlowAnalyzer:
    """数据流分析器"""
    def __init__(self):
        self.definitions: Dict[str, Set[int]] = {}  # 变量定义点
        self.uses: Dict[str, Set[int]] = {}        # 变量使用点
        self.live_vars: List[Set[str]] = []        # 活跃变量
        self.reaching_defs: List[Set[int]] = []    # 到达定义
        
    def analyze(self, instructions: List[Instruction]):
        """分析指令序列"""
        self.collect_defs_and_uses(instructions)
        self.compute_live_variables(instructions)
        self.compute_reaching_definitions(instructions)
        
    def collect_defs_and_uses(self, instructions: List[Instruction]):
        """收集定义和使用点"""
        for i, instr in enumerate(instructions):
            if instr.opcode == OpCode.STORE:
                var = instr.operand
                if var not in self.definitions:
                    self.definitions[var] = set()
                self.definitions[var].add(i)
                
            elif instr.opcode == OpCode.LOAD:
                var = instr.operand
                if var not in self.uses:
                    self.uses[var] = set()
                self.uses[var].add(i)
                
    def compute_live_variables(self, instructions: List[Instruction]):
        """计算活跃变量"""
        # 初始化活跃变量集
        self.live_vars = [set() for _ in range(len(instructions) + 1)]
        
        # 迭代计算直到不再变化
        changed = True
        while changed:
            changed = False
            
            # 从后向前遍历指令
            for i in range(len(instructions) - 1, -1, -1):
                instr = instructions[i]
                live = self.live_vars[i + 1].copy()
                
                # 处理指令的定义和使用
                if instr.opcode == OpCode.STORE:
                    live.discard(instr.operand)
                elif instr.opcode == OpCode.LOAD:
                    live.add(instr.operand)
                
                # 如果活跃集发生变化
                if live != self.live_vars[i]:
                    changed = True
                    self.live_vars[i] = live
                    
    def compute_reaching_definitions(self, instructions: List[Instruction]):
        """计算到达定义"""
        # 初始化到达定义集
        self.reaching_defs = [set() for _ in range(len(instructions) + 1)]
        
        # 迭代计算直到不再变化
        changed = True
        while changed:
            changed = False
            
            # 从前向后遍历指令
            for i, instr in enumerate(instructions):
                instr = instructions[i]
                reach = self.reaching_defs[i].copy()
                
                # 处理指令的定义
                if instr.opcode == OpCode.STORE:
                    # 删除变量的旧定义
                    var = instr.operand
                    reach = {d for d in reach if d not in self.definitions.get(var, set())}
                    # 添加新定义
                    reach.add(i)
                
                # 如果到达定义集发生变化
                if reach != self.reaching_defs[i + 1]:
                    changed = True
                    self.reaching_defs[i + 1] = reach

class Optimizer:
    """代码优化器"""
    def __init__(self, compiler):
        self.compiler = compiler
        self.scope_analyzer = ScopeAnalyzer()
        self.dataflow_analyzer = DataFlowAnalyzer()
        
    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        """执行所有优化"""
        if self.compiler.optimization_level >= 1:
            # 基本优化
            instructions = self.fold_constants(instructions)
            instructions = self.eliminate_dead_code(instructions)
            instructions = self.simplify_control_flow(instructions)
            
        if self.compiler.optimization_level >= 2:
            # 高级优化
            self.dataflow_analyzer.analyze(instructions)
            instructions = self.eliminate_dead_stores(instructions)
            instructions = self.common_subexpression_elimination(instructions)
            instructions = self.strength_reduction(instructions)
            instructions = self.loop_optimization(instructions)
            
        return instructions
        
    def fold_constants(self, instructions: List[Instruction]) -> List[Instruction]:
        """常量折叠"""
        result = []
        i = 0
        while i < len(instructions):
            if (i + 2 < len(instructions) and
                instructions[i].opcode == OpCode.LOAD_CONST and
                instructions[i + 1].opcode == OpCode.LOAD_CONST):
                
                const1 = self.compiler.constants[instructions[i].operand]
                const2 = self.compiler.constants[instructions[i + 1].operand]
                op = instructions[i + 2].opcode
                
                value = None
                if op == OpCode.ADD:
                    value = const1 + const2
                elif op == OpCode.SUB:
                    value = const1 - const2
                elif op == OpCode.MUL:
                    value = const1 * const2
                elif op == OpCode.DIV and const2 != 0:
                    value = const1 / const2
                elif op == OpCode.MOD and const2 != 0:
                    value = const1 % const2
                elif op == OpCode.POW:
                    value = const1 ** const2
                    
                if value is not None:
                    const_index = self.compiler.add_constant(value)
                    result.append(Instruction(OpCode.LOAD_CONST, const_index))
                    i += 3
                    continue
                    
            result.append(instructions[i])
            i += 1
            
        return result
        
    def eliminate_dead_stores(self, instructions: List[Instruction]) -> List[Instruction]:
        """消除死存储"""
        result = []
        for i, instr in enumerate(instructions):
            if instr.opcode == OpCode.STORE:
                var = instr.operand
                # 检查变量是否在后续被使用
                is_used = False
                for j in range(i + 1, len(instructions)):
                    if (instructions[j].opcode == OpCode.LOAD and 
                        instructions[j].operand == var):
                        is_used = True
                        break
                    if (instructions[j].opcode == OpCode.STORE and 
                        instructions[j].operand == var):
                        break
                
                if not is_used:
                    # 删除存储指令及其相关的计算指令
                    continue
                    
            result.append(instr)
            
        return result
        
    def common_subexpression_elimination(self, 
                                       instructions: List[Instruction]) -> List[Instruction]:
        """公共子表达式消除"""
        result = []
        expr_map = {}  # 表达式 -> 临时变量
        
        for i, instr in enumerate(instructions):
            if instr.opcode in [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV]:
                # 检查是否有相同的表达式已经计算过
                expr = (instr.opcode, result[-2].operand, result[-1].operand)
                if expr in expr_map:
                    # 使用之前计算的结果
                    temp_var = expr_map[expr]
                    result.pop()  # 移除操作数
                    result.pop()
                    result.append(Instruction(OpCode.LOAD, temp_var))
                else:
                    # 保存计算结果
                    temp_var = len(self.compiler.variables)
                    expr_map[expr] = temp_var
                    result.append(instr)
                    result.append(Instruction(OpCode.STORE, temp_var))
                continue
                
            result.append(instr)
            
        return result
        
    def strength_reduction(self, instructions: List[Instruction]) -> List[Instruction]:
        """强度削减"""
        result = []
        for i, instr in enumerate(instructions):
            if (instr.opcode == OpCode.MUL and 
                instructions[i - 1].opcode == OpCode.LOAD_CONST):
                # 将乘以2替换为左移
                const = self.compiler.constants[instructions[i - 1].operand]
                if const == 2:
                    result.pop()  # 移除常量加载
                    result.append(Instruction(OpCode.SHL, 1))
                    continue
                    
            elif (instr.opcode == OpCode.DIV and 
                  instructions[i - 1].opcode == OpCode.LOAD_CONST):
                # 将除以2替换为右移
                const = self.compiler.constants[instructions[i - 1].operand]
                if const == 2:
                    result.pop()  # 移除常量加载
                    result.append(Instruction(OpCode.SHR, 1))
                    continue
                    
            result.append(instr)
            
        return result
        
    def loop_optimization(self, instructions: List[Instruction]) -> List[Instruction]:
        """循环优化"""
        # 循环不变代码外提
        result = []
        loop_starts = []
        loop_ends = []
        
        # 找出所有循环
        for i, instr in enumerate(instructions):
            if instr.opcode == OpCode.JMP:
                if instr.operand < i:  # 向后跳转，是循环结尾
                    loop_starts.append(instr.operand)
                    loop_ends.append(i)
                    
        # 对每个循环进行优化
        for start, end in zip(loop_starts, loop_ends):
            loop_body = instructions[start:end + 1]
            
            # 找出循环不变的计算
            invariant_instrs = []
            for i, instr in enumerate(loop_body):
                if self.is_loop_invariant(instr, loop_body):
                    invariant_instrs.append(i)
                    
            # 将不变计算移到循环前
            for i in reversed(invariant_instrs):
                instr = loop_body.pop(i)
                result.insert(start, instr)
                
            result.extend(loop_body)
            
        return result
        
    def is_loop_invariant(self, instr: Instruction, 
                         loop_body: List[Instruction]) -> bool:
        """检查指令是否是循环不变的"""
        if instr.opcode == OpCode.LOAD_CONST:
            return True
            
        if instr.opcode == OpCode.LOAD:
            # 检查变量在循环中是否被修改
            var = instr.operand
            for other in loop_body:
                if other.opcode == OpCode.STORE and other.operand == var:
                    return False
            return True
            
        if instr.opcode in [OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV]:
            # 检查操作数是否都是循环不变的
            return (self.is_loop_invariant(loop_body[-2], loop_body) and
                   self.is_loop_invariant(loop_body[-1], loop_body))
                   
        return False
        
    def simplify_control_flow(self, instructions: List[Instruction]) -> List[Instruction]:
        """简化控制流"""
        result = []
        i = 0
        while i < len(instructions):
            if (instructions[i].opcode == OpCode.JZ and
                i + 1 < len(instructions) and
                instructions[i + 1].opcode == OpCode.JMP):
                # 将 JZ L1; JMP L2; L1: 转换为 JNZ L2
                result.append(Instruction(OpCode.JNZ, instructions[i + 1].operand))
                i += 2
                continue
                
            elif (instructions[i].opcode == OpCode.JMP and
                  i + 1 < len(instructions) and
                  instructions[i].operand == i + 1):
                # 删除跳转到下一条指令的JMP
                i += 1
                continue
                
            result.append(instructions[i])
            i += 1
            
        return result

class TypeAnalyzer:
    """类型分析器"""
    def __init__(self):
        self.type_info: Dict[str, str] = {}  # 变量 -> 类型
        self.type_constraints: List[Tuple[str, str]] = []  # (变量, 类型约束)
        self.type_errors: List[str] = []
        
    def analyze(self, node: ASTNode):
        """分析AST节点的类型"""
        if node.type == ASTNodeType.PROGRAM:
            for child in node.children:
                self.analyze(child)
                
        elif node.type == ASTNodeType.FUNCTION_DEFINITION:
            # 分析函数参数类型
            for param in node.params:
                param_type = self.get_param_type(param)
                self.type_info[param] = param_type
            
            # 分析函数体
            self.analyze(node.body)
            
            # 分析返回类型
            return_type = self.infer_return_type(node.body)
            self.type_info[node.value] = f"function({', '.join(self.type_info[p] for p in node.params)}) -> {return_type}"
            
        elif node.type == ASTNodeType.BINARY_OP:
            # 分析操作数类型
            left_type = self.analyze(node.children[0])
            right_type = self.analyze(node.children[1])
            
            # 检查类型兼容性
            result_type = self.check_operator_types(node.value, left_type, right_type)
            if result_type is None:
                self.type_errors.append(
                    f"Type error at line {node.line}: Cannot apply operator '{node.value}' "
                    f"to types '{left_type}' and '{right_type}'"
                )
            return result_type
            
        elif node.type == ASTNodeType.FUNCTION_CALL or node.type == ASTNodeType.TA_FUNCTION:
            # 分析函数调用
            if node.value in self.type_info:
                func_type = self.type_info[node.value]
                if not func_type.startswith("function"):
                    self.type_errors.append(
                        f"Type error at line {node.line}: '{node.value}' is not a function"
                    )
                    return "error"
                    
                # 检查参数数量和类型
                param_types = self.get_function_param_types(func_type)
                if len(param_types) != len(node.children):
                    self.type_errors.append(
                        f"Type error at line {node.line}: Function '{node.value}' expects "
                        f"{len(param_types)} arguments, but got {len(node.children)}"
                    )
                    return "error"
                    
                for i, (param_type, arg) in enumerate(zip(param_types, node.children)):
                    arg_type = self.analyze(arg)
                    if not self.is_type_compatible(param_type, arg_type):
                        self.type_errors.append(
                            f"Type error at line {node.line}: In call to '{node.value}', "
                            f"argument {i+1} has type '{arg_type}' but expected '{param_type}'"
                        )
                        
                return self.get_function_return_type(func_type)
            else:
                # 处理内置函数
                return self.analyze_builtin_call(node)
                
        elif node.type == ASTNodeType.ARRAY_LITERAL:
            # 分析数组字面量
            if not node.children:
                return "array<any>"
                
            element_type = self.analyze(node.children[0])
            for child in node.children[1:]:
                child_type = self.analyze(child)
                if not self.is_type_compatible(element_type, child_type):
                    self.type_errors.append(
                        f"Type error at line {node.line}: Array elements must have compatible types, "
                        f"found '{element_type}' and '{child_type}'"
                    )
                    return "error"
                    
            return f"array<{element_type}>"
            
        elif node.type == ASTNodeType.ARRAY_METHOD:
            # 分析数组访问
            array_type = self.analyze(node.children[0])
            if not array_type.startswith("array<"):
                self.type_errors.append(
                    f"Type error at line {node.line}: Cannot index non-array type '{array_type}'"
                )
                return "error"
                
            index_type = self.analyze(node.children[1])
            if index_type != "int":
                self.type_errors.append(
                    f"Type error at line {node.line}: Array index must be integer, got '{index_type}'"
                )
                
            return array_type[6:-1]  # 返回数组元素类型
            
        elif node.type == ASTNodeType.ASSIGNMENT:
            # 分析赋值语句
            value_type = self.analyze(node.value)
            if hasattr(node, 'type') and node.type == ASTNodeType.ARRAY_METHOD:
                array_type = self.analyze(node.children[0])
                if not array_type.startswith("array<"):
                    self.type_errors.append(
                        f"Type error at line {node.line}: Cannot index non-array type '{array_type}'"
                    )
                    return "error"
                    
                element_type = array_type[6:-1]
                if not self.is_type_compatible(element_type, value_type):
                    self.type_errors.append(
                        f"Type error at line {node.line}: Cannot assign value of type '{value_type}' "
                        f"to array element of type '{element_type}'"
                    )
            else:
                if node.target in self.type_info:
                    target_type = self.type_info[node.target]
                    if not self.is_type_compatible(target_type, value_type):
                        self.type_errors.append(
                            f"Type error at line {node.line}: Cannot assign value of type '{value_type}' "
                            f"to variable of type '{target_type}'"
                        )
                else:
                    self.type_info[node.target] = value_type
                    
            return value_type
            
        elif node.type == ASTNodeType.NUMBER:
            return "float" if isinstance(node.value, float) else "int"
            
        elif node.type == ASTNodeType.STRING:
            return "string"
            
        elif node.type == ASTNodeType.IDENTIFIER:
            if node.value == "true" or node.value == "false":
                return "bool"
            if node.value not in self.type_info:
                self.type_errors.append(
                    f"Type error at line {node.line}: Undefined variable '{node.value}'"
                )
                return "error"
            return self.type_info[node.value]
            
        elif node.type == ASTNodeType.PLOT_STATEMENT:
            # 分析绘图语句
            if node.children:
                series_type = self.analyze(node.children[0])
                if series_type not in ['int', 'float']:
                    self.type_errors.append(
                        f"Type error at line {node.line}: Plot function expects numeric series, "
                        f"got '{series_type}'"
                    )
            return "void"
            
        elif node.type == ASTNodeType.RETURN_STMT:
            # 分析返回语句
            if node.children:
                return self.analyze(node.children[0])
            return "void"
            
        return "void"

    def check_operator_types(self, operator: str, left_type: str, right_type: str) -> Optional[str]:
        """检查操作符类型兼容性并返回结果类型"""
        # 算术运算符
        if operator in ['+', '-', '*', '/', '%']:
            if left_type in ['int', 'float'] and right_type in ['int', 'float']:
                return 'float' if 'float' in [left_type, right_type] else 'int'
            if operator == '+' and 'string' in [left_type, right_type]:
                return 'string'
                
        # 比较运算符
        elif operator in ['<', '>', '<=', '>=']:
            if left_type in ['int', 'float'] and right_type in ['int', 'float']:
                return 'bool'
            if left_type == right_type == 'string':
                return 'bool'
                
        # 相等运算符
        elif operator in ['==', '!=']:
            if left_type == right_type:
                return 'bool'
                
        # 逻辑运算符
        elif operator in ['&&', '||']:
            if left_type == right_type == 'bool':
                return 'bool'
                
        # 位运算符
        elif operator in ['&', '|', '^', '<<', '>>']:
            if left_type == right_type == 'int':
                return 'int'
                
        return None

    def is_type_compatible(self, target_type: str, source_type: str) -> bool:
        """检查类型是否兼容"""
        if target_type == source_type:
            return True
            
        if target_type == 'float' and source_type == 'int':
            return True
            
        if target_type.startswith('array<') and source_type.startswith('array<'):
            return self.is_type_compatible(
                target_type[6:-1],  # 内部类型
                source_type[6:-1]
            )
            
        return False

    def analyze_builtin_call(self, node: ASTNode) -> str:
        """分析内置函数调用"""
        if node.value in ['sma', 'ema', 'rsi', 'macd']:
            # 检查参数类型
            for arg in node.children:
                arg_type = self.analyze(arg)
                if arg_type not in ['int', 'float']:
                    self.type_errors.append(
                        f"Type error at line {node.line}: Technical analysis function '{node.value}' "
                        f"expects numeric arguments, got '{arg_type}'"
                    )
            return 'float'
            
        elif node.value in ['cross', 'crossover', 'crossunder']:
            if len(node.children) != 2:
                self.type_errors.append(
                    f"Type error at line {node.line}: Function '{node.value}' expects 2 arguments"
                )
                return 'error'
                
            for arg in node.children:
                arg_type = self.analyze(arg)
                if arg_type not in ['int', 'float']:
                    self.type_errors.append(
                        f"Type error at line {node.line}: Function '{node.value}' "
                        f"expects numeric arguments, got '{arg_type}'"
                    )
            return 'bool'
            
        elif node.value.startswith('plot'):
            # 检查绘图函数的参数
            series_type = self.analyze(node.children[0])
            if series_type not in ['int', 'float']:
                self.type_errors.append(
                    f"Type error at line {node.line}: Plot function expects numeric series, "
                    f"got '{series_type}'"
                )
            return 'void'
            
        elif node.value.startswith('strategy'):
            # 检查策略函数的参数
            if hasattr(node, 'condition'):
                cond_type = self.analyze(node.condition)
                if cond_type != 'bool':
                    self.type_errors.append(
                        f"Type error at line {node.line}: Strategy condition must be boolean, "
                        f"got '{cond_type}'"
                    )
                    
            if hasattr(node, 'quantity'):
                qty_type = self.analyze(node.quantity)
                if qty_type not in ['int', 'float']:
                    self.type_errors.append(
                        f"Type error at line {node.line}: Strategy quantity must be numeric, "
                        f"got '{qty_type}'"
                    )
            return 'void'
            
        return 'any'  # 其他内置函数

    def get_param_type(self, param: str) -> str:
        """获取参数类型（可以从注解或上下文推断）"""
        # 这里可以添加更复杂的类型推断逻辑
        return "any"

    def infer_return_type(self, body: ASTNode) -> str:
        """推断函数返回类型"""
        # 分析return语句
        return_types = set()
        
        def collect_return_types(node):
            if node.type == ASTNodeType.RETURN_STMT:
                if node.children:
                    return_type = self.analyze(node.children[0])
                    return_types.add(return_type)
                else:
                    return_types.add("void")
            for child in node.children:
                collect_return_types(child)
                
        collect_return_types(body)
        
        if not return_types:
            return "void"
        if len(return_types) == 1:
            return next(iter(return_types))
        # 如果有多个返回类型，选择最通用的类型
        if "error" in return_types:
            return "error"
        if "any" in return_types:
            return "any"
        if "float" in return_types and "int" in return_types:
            return "float"
        return "any"

    def get_function_param_types(self, func_type: str) -> List[str]:
        """从函数类型字符串中提取参数类型"""
        if not func_type.startswith("function("):
            return []
        params_str = func_type[9:func_type.index(")")]
        return [p.strip() for p in params_str.split(",")] if params_str else []

    def get_function_return_type(self, func_type: str) -> str:
        """从函数类型字符串中提取返回类型"""
        return func_type[func_type.index("->") + 2:].strip()

class CompilerDiagnostics:
    """编译器诊断信息"""
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.type_errors: List[str] = []
        self.optimization_info: List[str] = []
        
    def add_error(self, message: str, line: int, column: int):
        """添加错误信息"""
        self.errors.append(f"Error at line {line}, column {column}: {message}")
        
    def add_warning(self, message: str, line: int, column: int):
        """添加警告信息"""
        self.warnings.append(f"Warning at line {line}, column {column}: {message}")
        
    def add_info(self, message: str):
        """添加信息"""
        self.info.append(message)
        
    def add_type_error(self, message: str, line: int, column: int):
        """添加类型错误信息"""
        self.type_errors.append(f"Type error at line {line}, column {column}: {message}")
        
    def add_optimization_info(self, message: str):
        """添加优化信息"""
        self.optimization_info.append(message)
        
    def has_errors(self) -> bool:
        """检查是否有错误"""
        return bool(self.errors or self.type_errors)
        
    def format_messages(self) -> str:
        """格式化所有消息"""
        messages = []
        if self.errors:
            messages.append("Errors:")
            messages.extend(f"  {error}" for error in self.errors)
            
        if self.type_errors:
            messages.append("Type Errors:")
            messages.extend(f"  {error}" for error in self.type_errors)
            
        if self.warnings:
            messages.append("Warnings:")
            messages.extend(f"  {warning}" for warning in self.warnings)
            
        if self.optimization_info:
            messages.append("Optimization Info:")
            messages.extend(f"  {info}" for info in self.optimization_info)
            
        if self.info:
            messages.append("Additional Info:")
            messages.extend(f"  {info}" for info in self.info)
            
        return "\n".join(messages)
