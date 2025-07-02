"""
Rosenbrock函数详细实现原理解析
=====================================

Rosenbrock函数是优化领域最经典的测试函数之一，被称为"香蕉函数"，
因为其等高线图形状像香蕉。它具有以下特点：

1. 全局最优解位于狭窄的抛物线山谷中
2. 算法容易找到山谷，但很难收敛到谷底
3. 测试算法的精确搜索能力

数学表达式：
f(x) = Σ[100*(x_i² - x_{i+1})² + (x_i - 1)²] for i=1 to n-1

全局最优解：x* = (1, 1, ..., 1)，f(x*) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simple_rosenbrock(x):
    """
    简化版Rosenbrock函数 - 便于理解基本原理
    """
    f = 0.0
    n = len(x)
    
    for i in range(n - 1):
        # 第一项：100*(x_i² - x_{i+1})²
        # 这项确保相邻变量满足 x_{i+1} ≈ x_i²
        term1 = 100.0 * (x[i]**2 - x[i+1])**2
        
        # 第二项：(x_i - 1)²  
        # 这项将最优解推向 x_i = 1
        term2 = (x[i] - 1.0)**2
        
        f += term1 + term2
        
    return f

def cec_rosenbrock(x, nx, Os, Mr, s_flag, r_flag):
    """
    CEC版本的Rosenbrock函数 - 包含变换
    
    变换步骤：
    1. 位移变换：x' = x - Os
    2. 旋转变换：x'' = Mr * x'  
    3. 缩放变换：x''' = x'' * (2.048/100)
    4. 最优解调整：x''' = x''' + 1
    """
    
    # 第1步：sr_func进行组合变换
    z = sr_func(x, nx, Os, Mr, 2.048/100.0, s_flag, r_flag)
    
    # 第2步：调整使最优解在z=(1,1,...,1)
    z[0] += 1.0
    
    f = 0.0
    for i in range(nx - 1):
        z[i + 1] += 1.0
        
        # Rosenbrock标准公式
        tmp1 = z[i] * z[i] - z[i + 1]    # x_i² - x_{i+1}
        tmp2 = z[i] - 1.0                # x_i - 1
        f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        
    return f

def visualize_rosenbrock_2d():
    """
    可视化2D Rosenbrock函数
    """
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # 计算函数值
    Z = 100 * (X**2 - Y)**2 + (X - 1)**2
    
    # 绘制等高线图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    contour = plt.contour(X, Y, Z, levels=50)
    plt.colorbar(contour)
    plt.plot(1, 1, 'r*', markersize=15, label='全局最优解 (1,1)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Rosenbrock函数等高线图')
    plt.legend()
    
    # 绘制3D表面图
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.scatter([1], [1], [0], color='red', s=100, label='全局最优解')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.set_title('Rosenbrock函数3D图')
    plt.colorbar(surf)
    
    plt.tight_layout()
    plt.savefig('rosenbrock_visualization.png', dpi=300)
    plt.show()

def analyze_difficulty():
    """
    分析Rosenbrock函数的优化难点
    """
    print("=== Rosenbrock函数优化难点分析 ===")
    print()
    
    print("1. 山谷特性：")
    print("   - 全局最优解位于抛物线 x2 = x1² 上")
    print("   - 山谷很窄，算法容易偏离")
    print("   - 山谷内梯度很小，收敛缓慢")
    print()
    
    print("2. 梯度分析：")
    print("   ∇f = [400*x1*(x1² - x2) + 2*(x1 - 1)]")
    print("        [200*(x2 - x1²)]")
    print("   - 在山谷底部梯度接近0")
    print("   - 不同方向梯度变化剧烈")
    print()
    
    print("3. 条件数问题：")
    print("   - Hessian矩阵条件数很大")
    print("   - 导致数值优化困难")
    print("   - 需要自适应步长策略")

if __name__ == "__main__":
    # 测试简单版本
    x_test = [0.5, 0.5]
    result = simple_rosenbrock(x_test)
    print(f"简单Rosenbrock函数测试: f({x_test}) = {result}")
    
    # 分析优化难点
    analyze_difficulty()
    
    # 可视化 (如果环境支持)
    try:
        visualize_rosenbrock_2d()
    except:
        print("无法生成可视化图形，请确保安装matplotlib") 