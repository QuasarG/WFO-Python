"""
自定义测试函数完整指南
========================

本指南演示如何创建新的测试函数并集成到WFO项目中。
包括：函数设计、参数设置、数据准备、集成方法等。
"""

import numpy as np
from math import pi, sin, cos, exp, sqrt, log
import os

class CustomTestFunction:
    """
    自定义测试函数基类
    
    所有自定义函数都应该继承这个类，并实现核心方法
    """
    
    def __init__(self, name, dimension_range, search_range, global_optimum):
        """
        初始化自定义函数
        
        参数:
            name: 函数名称
            dimension_range: 支持的维度范围 [min_dim, max_dim]
            search_range: 搜索范围 [lower_bound, upper_bound]
            global_optimum: 全局最优值
        """
        self.name = name
        self.dimension_range = dimension_range
        self.search_range = search_range
        self.global_optimum = global_optimum
    
    def evaluate(self, x):
        """
        函数评估 - 子类必须重写此方法
        """
        raise NotImplementedError("子类必须实现evaluate方法")
    
    def get_optimal_solution(self, dimension):
        """
        获取指定维度下的最优解 - 子类可选择重写
        """
        return [0.0] * dimension  # 默认最优解在原点

# === 示例1：简单单峰函数 ===
class MySphereFunction(CustomTestFunction):
    """
    自定义球函数变种
    f(x) = Σ(w_i * x_i²) + bias
    """
    
    def __init__(self):
        super().__init__(
            name="自定义加权球函数",
            dimension_range=[2, 100], 
            search_range=[-10, 10],
            global_optimum=5.0  # 加了偏置
        )
        
    def evaluate(self, x):
        """计算加权球函数值"""
        n = len(x)
        f = 0.0
        
        for i in range(n):
            # 不同维度不同权重
            weight = 1.0 + 0.1 * i
            f += weight * x[i] * x[i]
        
        # 添加偏置，使最优值不为0
        f += 5.0
        return f

# === 示例2：复杂多峰函数 ===
class MyMultiModalFunction(CustomTestFunction):
    """
    自定义多峰函数
    f(x) = Σ(x_i² - 10*cos(2π*x_i)) + 10*n + noise
    """
    
    def __init__(self):
        super().__init__(
            name="改进Rastrigin函数",
            dimension_range=[2, 50],
            search_range=[-5.12, 5.12], 
            global_optimum=0.0
        )
        
    def evaluate(self, x):
        """计算改进的Rastrigin函数值"""
        n = len(x)
        f = 0.0
        
        for i in range(n):
            # 基本Rastrigin项
            f += x[i]**2 - 10*cos(2*pi*x[i])
            
            # 添加高频振荡 
            f += 0.5 * sin(10*pi*x[i])
        
        f += 10 * n
        return f

# === 示例3：混合函数 ===  
class MyHybridFunction(CustomTestFunction):
    """
    自定义混合函数 - 结合多个基础函数
    """
    
    def __init__(self):
        super().__init__(
            name="自定义混合函数",
            dimension_range=[4, 20],
            search_range=[-10, 10],
            global_optimum=0.0
        )
    
    def sphere_part(self, x):
        """球函数部分"""
        return sum(xi**2 for xi in x)
    
    def rosenbrock_part(self, x):
        """Rosenbrock函数部分"""
        f = 0.0
        for i in range(len(x) - 1):
            f += 100*(x[i]**2 - x[i+1])**2 + (x[i] - 1)**2
        return f
    
    def ackley_part(self, x):
        """Ackley函数部分"""
        n = len(x)
        sum1 = sum(xi**2 for xi in x)
        sum2 = sum(cos(2*pi*xi) for xi in x)
        return -20*exp(-0.2*sqrt(sum1/n)) - exp(sum2/n) + 20 + exp(1)
    
    def evaluate(self, x):
        """计算混合函数值"""
        n = len(x)
        
        # 将维度分为三部分
        part_size = n // 3
        
        # 第一部分：球函数
        x1 = x[:part_size]
        f1 = self.sphere_part(x1) if x1 else 0
        
        # 第二部分：Rosenbrock函数  
        x2 = x[part_size:2*part_size]
        f2 = self.rosenbrock_part(x2) if len(x2) > 1 else 0
        
        # 第三部分：Ackley函数
        x3 = x[2*part_size:]
        f3 = self.ackley_part(x3) if x3 else 0
        
        # 加权组合
        return 0.3*f1 + 0.4*f2 + 0.3*f3

# === 示例4：非分离函数 ===
class MyNonSeparableFunction(CustomTestFunction):
    """
    自定义非分离函数 - 变量间强相关
    """
    
    def __init__(self):
        super().__init__(
            name="自定义非分离函数", 
            dimension_range=[2, 30],
            search_range=[-5, 5],
            global_optimum=0.0
        )
    
    def evaluate(self, x):
        """计算非分离函数值"""
        n = len(x)
        f = 0.0
        
        # 主对角线项
        for i in range(n):
            f += x[i]**2
        
        # 非分离项：相邻变量相关
        for i in range(n-1):
            f += 0.5 * x[i] * x[i+1]
        
        # 全局相关项  
        sum_x = sum(x)
        f += 0.1 * sum_x**2
        
        # 添加三角函数项增加复杂性
        for i in range(n):
            for j in range(i+1, n):
                f += 0.01 * sin(x[i] + x[j])
                
        return f

def integrate_custom_function_to_demo():
    """
    演示如何将自定义函数集成到Demo.py中
    """
    
    integration_code = '''
# === 在Demo.py中集成自定义函数的方法 ===

from WFO import WFO
from matplotlib import pyplot
from 自定义测试函数指南 import MySphereFunction, MyMultiModalFunction

# 创建自定义函数实例
custom_func = MySphereFunction()

# 设置算法参数
class alg:
    NP = 30
    max_nfe = 15000
    pl = 0.7
    pe = 0.3
    
# 设置问题参数
n = 10   # 维度
class prob:
    dim = n
    lb = [custom_func.search_range[0]] * n  # 使用函数的搜索范围
    ub = [custom_func.search_range[1]] * n
    
    # 使用自定义函数
    def fobj(x):        
        return custom_func.evaluate(x)

# 运行优化
fb, xb, con = WFO(alg, prob)

print(f'函数名称: {custom_func.name}')
print(f'最小目标函数值: {fb}')
print(f'理论最优值: {custom_func.global_optimum}')
print(f'优化误差: {abs(fb - custom_func.global_optimum)}')
print(f'最优解: {xb}')

# 绘制收敛曲线
pyplot.plot(con)
pyplot.xlabel('函数评估次数')
pyplot.ylabel('函数值')
pyplot.title(f'{custom_func.name} - 收敛曲线')
pyplot.yscale('log')  # 对数坐标
pyplot.show()
'''
    
    print("=== 自定义函数集成示例 ===")
    print(integration_code)

def create_function_test_suite():
    """
    创建函数测试套件 - 批量测试多个自定义函数
    """
    
    test_suite_code = '''
# === 自定义函数测试套件 ===

import numpy as np
from WFO import WFO

def test_custom_functions():
    """批量测试所有自定义函数"""
    
    # 创建函数列表
    functions = [
        MySphereFunction(),
        MyMultiModalFunction(), 
        MyHybridFunction(),
        MyNonSeparableFunction()
    ]
    
    # 测试维度
    test_dimensions = [5, 10, 20]
    
    # 算法参数
    class alg:
        NP = 50
        max_nfe = 20000
        pl = 0.7
        pe = 0.3
    
    results = []
    
    for func in functions:
        for dim in test_dimensions:
            # 检查维度是否支持
            if dim < func.dimension_range[0] or dim > func.dimension_range[1]:
                continue
                
            # 设置问题
            class prob:
                dim = dim
                lb = [func.search_range[0]] * dim
                ub = [func.search_range[1]] * dim
                fobj = func.evaluate
            
            # 运行优化
            print(f"测试 {func.name} - {dim}维...")
            fb, xb, con = WFO(alg, prob)
            
            # 计算误差
            error = abs(fb - func.global_optimum)
            
            # 记录结果
            results.append({
                'function': func.name,
                'dimension': dim, 
                'best_value': fb,
                'optimal_value': func.global_optimum,
                'error': error,
                'solution': xb
            })
            
            print(f"  最优值: {fb:.6f}, 误差: {error:.6e}")
    
    return results

# 运行测试
if __name__ == "__main__":
    test_results = test_custom_functions()
    
    # 输出汇总报告
    print("\\n=== 测试结果汇总 ===")
    for result in test_results:
        print(f"{result['function']} ({result['dimension']}D): "
              f"误差 = {result['error']:.2e}")
'''
    
    print("=== 函数测试套件代码 ===")
    print(test_suite_code)

def extend_cec_framework():
    """
    演示如何扩展CEC框架添加新函数
    """
    
    extension_code = '''
# === 扩展CEC2022框架的方法 ===

# 1. 在CEC2022.py中添加新函数

def my_custom_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    新的CEC风格函数
    
    参数:
        x: 输入向量
        nx: 维度
        Os: 位移向量  
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    """
    # 应用标准变换
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    
    # 计算自定义函数值
    f = 0.0
    for i in range(nx):
        f += z[i]**4 - 16*z[i]**2 + 5*z[i]  # 四次多项式
        
    return f

# 2. 修改cec22_test_func函数

def cec22_test_func(x, nx, mx, func_num):
    # ... 现有代码 ...
    
    # 在函数选择部分添加新函数
    for i in range(mx):
        if func_num == 13:  # 新函数编号
            ff = my_custom_func(x, nx, OShift, M, 1, 1)
            f[i] = ff + 3000.0  # 添加偏移
            break
        # ... 其他函数 ...
    
    return f

# 3. 准备数据文件

def prepare_data_files():
    """为新函数准备数据文件"""
    
    # 创建位移数据
    for dim in [2, 10, 20]:
        shift_data = np.random.uniform(-80, 80, dim)
        np.savetxt(f'input_data/shift_data_13.txt', shift_data)
    
    # 创建旋转矩阵
    for dim in [2, 10, 20]:
        # 生成正交随机矩阵
        A = np.random.randn(dim, dim)
        Q, R = np.linalg.qr(A)
        np.savetxt(f'input_data/M_13_D{dim}.txt', Q)

# 4. 在WFO_exe.py GUI中添加新函数

# 修改函数列表
funcChosen['values'] = (1,2,3,4,5,6,7,8,9,10,11,12,13)  # 添加13

# 修改enter()函数处理新函数
def enter():
    num = int(funcChosen.get())
    if num == 13:
        fitFuncEntered.insert(0, 'cec22_test_func(x = x, nx = n, mx = 1, func_num = 13)')
    # ... 其他处理 ...
'''
    
    print("=== CEC框架扩展代码 ===")
    print(extension_code)

if __name__ == "__main__":
    print("=== 自定义测试函数完整指南 ===\n")
    
    # 测试所有自定义函数
    functions = [
        MySphereFunction(),
        MyMultiModalFunction(),
        MyHybridFunction(), 
        MyNonSeparableFunction()
    ]
    
    print("1. 自定义函数测试:")
    test_x = [1.0, 2.0, -1.0, 0.5]
    
    for func in functions:
        result = func.evaluate(test_x)
        print(f"   {func.name}: f({test_x}) = {result:.6f}")
    
    print("\n2. 函数属性:")
    for func in functions:
        print(f"   {func.name}:")
        print(f"     - 维度范围: {func.dimension_range}")
        print(f"     - 搜索范围: {func.search_range}")  
        print(f"     - 全局最优值: {func.global_optimum}")
    
    print("\n3. 集成方法:")
    integrate_custom_function_to_demo()
    
    print("\n4. 测试套件:")
    create_function_test_suite()
    
    print("\n5. CEC框架扩展:")
    extend_cec_framework() 