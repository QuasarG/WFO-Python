# -*- coding: utf-8 -*-
"""
CEC2022基准测试函数集
====================

本模块实现了CEC2022竞赛中的12个基准测试函数，包括：
- 基础函数 (F1-F5): 单峰和多峰经典函数
- 混合函数 (F6-F8): 组合多个基础函数的混合优化问题  
- 组合函数 (F9-F12): 加权组合多个函数形成复杂景观

主要特点：
1. 支持2维、10维、20维测试
2. 包含位移、旋转、缩放等变换
3. 提供标准化的测试接口
4. 自动加载预设的测试数据

Created on Sat Jan  1 16:49:21 2022
@author: Abhishek Kumar
@email: abhishek.kumar.eee13@iitbhu.ac.in
"""
import numpy as np

# 全局常量定义
INF = 1.0e99                    # 无穷大值，用于异常处理
EPS = 1.0e-14                   # 极小值，用于数值计算精度控制
E = 2.7182818284590452353602874713526625    # 自然常数e
PI = 3.1415926535897932384626433832795029   # 圆周率π


def ellips_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    椭球函数 (Ellipsoid Function)
    
    数学表达式: f(x) = Σ(10^(6i/(n-1)) * z_i^2)
    特点：单峰函数，在不同维度上具有不同的敏感度
    
    参数:
        x: 输入向量 
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志 (1=执行位移, 0=不位移)
        r_flag: 旋转标志 (1=执行旋转, 0=不旋转)
    
    返回:
        f: 函数值
    """
    f = 0.0
    # 应用位移和旋转变换
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    
    # 计算椭球函数值，每维度具有不同的权重
    for i in range(nx):
        f += pow(10.0, 6.0 * i / (nx - 1)) * z[i] * z[i]
    return f


def bent_cigar_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    弯曲雪茄函数 (Bent Cigar Function)
    
    数学表达式: f(x) = z_1^2 + 10^6 * Σ(z_i^2) for i=2 to n
    特点：第一维权重为1，其余维度权重为10^6，形成细长的椭球
    
    参数:
        x: 输入向量
        nx: 问题维度  
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    f = z[0] * z[0]  # 第一维正常权重
    
    # 其余维度使用高权重
    for i in range(1, nx):
        f += pow(10.0, 6.0) * z[i] * z[i]
    return f


def discus_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    铁饼函数 (Discus Function)
    
    数学表达式: f(x) = 10^6 * z_1^2 + Σ(z_i^2) for i=2 to n
    特点：第一维权重为10^6，其余维度权重为1，与弯曲雪茄相反
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量  
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    f = pow(10.0, 6.0) * z[0] * z[0]  # 第一维高权重
    
    # 其余维度正常权重
    for i in range(1, nx):
        f += z[i] * z[i]
    return f


def rosenbrock_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Rosenbrock函数 (Rosenbrock Function)
    
    数学表达式: f(x) = Σ(100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2)
    特点：经典的非凸优化函数，具有香蕉形状的山谷，全局最优解在狭窄的抛物线山谷中
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    f = 0.0
    # 特殊缩放：2.048/100.0，将搜索范围压缩
    z = sr_func(x, nx, Os, Mr, 2.048 / 100.0, s_flag, r_flag)
    z[0] += 1.0  # 移动最优解位置
    
    for i in range(nx - 1):
        z[i + 1] += 1.0
        tmp1 = z[i] * z[i] - z[i + 1]    # 相邻变量的二次关系
        tmp2 = z[i] - 1.0                # 偏离最优解的距离
        f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2  # Rosenbrock标准公式
    return f


def ackley_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Ackley函数 (Ackley Function)
    
    数学表达式: f(x) = -20*exp(-0.2*sqrt(mean(z_i^2))) - exp(mean(cos(2π*z_i))) + 20 + e
    特点：多峰函数，具有许多局部最优解，全局最优解为0
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵  
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    sum1, sum2 = 0, 0
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    
    for i in range(nx):
        sum1 += z[i] * z[i]              # 二次项求和
        sum2 += np.cos(2.0 * PI * z[i])  # 余弦项求和，产生多峰特性
    
    sum1 = -0.2 * np.sqrt(sum1 / nx)    # 指数项的指数部分
    sum2 /= nx                           # 余弦项平均值
    
    # Ackley函数的标准公式
    f = E - 20.0 * np.exp(sum1) - np.exp(sum2) + 20.0
    return f


def griewank_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Griewank函数 (Griewank Function)
    
    数学表达式: f(x) = 1 + Σ(z_i^2)/4000 - Π(cos(z_i/sqrt(i+1)))
    特点：多峰函数，远离原点时主要由二次项主导，近原点时余弦项产生波动
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    s = 0.0   # 二次项累加
    p = 1.0   # 余弦项累乘
    
    # 特殊缩放：600.0/100.0，调整函数的尺度
    z = sr_func(x, nx, Os, Mr, 600.0 / 100.0, s_flag, r_flag)
    
    for i in range(nx):
        s += z[i] * z[i]                           # 二次项
        p *= np.cos(z[i] / np.sqrt(1.0 + i))       # 余弦乘积项，频率随维度变化
    
    f = 1.0 + s / 4000.0 - p
    return f


def rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Rastrigin函数 (Rastrigin Function)
    
    数学表达式: f(x) = Σ(z_i^2 - 10*cos(2π*z_i) + 10)
    特点：高度多峰函数，具有大量局部最优解，是测试全局优化算法的经典函数
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    f = 0.0
    # 缩放到[-5.12, 5.12]范围
    z = sr_func(x, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag)
    
    for i in range(nx):
        # Rastrigin函数：二次项 + 余弦振荡项
        f += (z[i] * z[i] - 10.0 * np.cos(2.0 * PI * z[i]) + 10.0)
    return f


def schwefel_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Schwefel函数 (Schwefel Function)
    
    数学表达式: f(x) = 418.9829*n - Σ(z_i * sin(sqrt(|z_i|)))
    特点：复杂多峰函数，全局最优解远离局部最优解，具有欺骗性
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    f = 0.0
    # 缩放到[-500, 500]范围
    z = sr_func(x, nx, Os, Mr, 1000.0 / 100.0, s_flag, r_flag)
    
    for i in range(nx):
        z[i] += 4.209687462275036e+002  # 位移到最优解位置
        
        # 边界处理：超出[-500, 500]范围时的特殊处理
        if z[i] > 500:
            f -= (500.0 - np.fmod(z[i], 500)) * np.sin(pow(500.0 - np.fmod(z[i], 500), 0.5))
            tmp = (z[i] - 500.0) / 100
            f += tmp * tmp / nx
        elif (z[i] < -500):
            f -= (-500.0 + np.fmod(np.fabs(z[i]), 500)) * np.sin(pow(500.0 - np.fmod(np.fabs(z[i]), 500), 0.5))
            tmp = (z[i] + 500.0) / 100
            f += tmp * tmp / nx
        else:
            # 正常范围内的Schwefel函数
            f -= z[i] * np.sin(pow(np.fabs(z[i]), 0.5))
    
    f += 4.189828872724338e+002 * nx  # 加上常数项使最优值接近0
    return f


def grie_rosen_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Griewank-Rosenbrock函数 (Expanded Griewank plus Rosenbrock Function)
    
    特点：结合了Griewank和Rosenbrock函数的特性，既有多峰性又有非凸性
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    f = 0.0
    z = sr_func(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
    
    z[0] += 1.0
    for i in range(nx - 1):
        z[i + 1] += 1.0
        # 先计算Rosenbrock项
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        
        # 再应用Griewank变换
        f += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
    
    # 处理最后一维的环形连接
    tmp1 = z[nx - 1] * z[nx - 1] - z[0]
    tmp2 = z[nx - 1] - 1.0
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    f += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
    return f


def escaffer6_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    扩展Schaffer F6函数 (Expanded Schaffer's F6 Function)
    
    数学表达式: f(x) = Σ(0.5 + (sin^2(sqrt(z_i^2 + z_{i+1}^2)) - 0.5) / (1 + 0.001*(z_i^2 + z_{i+1}^2))^2)
    特点：多峰函数，具有许多接近的局部最优解
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    f = 0.0
    
    # 相邻维度对的处理
    for i in range(nx - 1):
        temp1 = np.sin(np.sqrt(z[i] * z[i] + z[i + 1] * z[i + 1]))
        temp1 = temp1 * temp1
        temp2 = 1.0 + 0.001 * (z[i] * z[i] + z[i + 1] * z[i + 1])
        f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    
    # 最后一维与第一维的环形连接
    temp1 = np.sin(np.sqrt(z[nx - 1] * z[nx - 1] + z[0] * z[0]))
    temp1 = temp1 * temp1
    temp2 = 1.0 + 0.001 * (z[nx - 1] * z[nx - 1] + z[0] * z[0])
    f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    return f


def happycat_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Happy Cat函数
    
    数学表达式: f(x) = |r^2 - n|^(2α) + (0.5*r^2 + Σz_i)/n + 0.5
    特点：具有连续的梯度，适合测试算法的收敛性
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    alpha = 1.0 / 8.0  # 幂指数参数
    z = sr_func(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
    
    r2 = 0.0      # 半径的平方
    sum_z = 0.0   # 坐标和
    
    for i in range(nx):
        z[i] = z[i] - 1.0    # 位移使最优解在z=(1,1,...,1)
        r2 += z[i] * z[i]    
        sum_z += z[i]
    
    # Happy Cat函数公式
    f = pow(np.fabs(r2 - nx), 2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5
    return f


def hgbat_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    HGBat函数 (HGBat Function)
    
    数学表达式: f(x) = |(r^2)^2 - (Σz_i)^2|^(2α) + (0.5*r^2 + Σz_i)/n + 0.5
    特点：比Happy Cat函数更复杂，具有更多的局部最优解
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    alpha = 1.0 / 4.0  # 幂指数参数
    z = sr_func(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
    
    r2 = 0.0      # 半径的平方
    sum_z = 0.0   # 坐标和
    
    for i in range(nx):
        z[i] = z[i] - 1.0    # 位移使最优解在z=(1,1,...,1)
        r2 += z[i] * z[i]
        sum_z += z[i]
    
    # HGBat函数公式：比Happy Cat多了一个平方项
    f = pow(np.fabs(pow(r2, 2.0) - pow(sum_z, 2.0)), 2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5
    return f


def schaffer_F7_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Schaffer's F7函数 (Schaffer's F7 Function)
    
    数学表达式: f(x) = (1/(n-1) * Σ(sqrt(z_i^2 + z_{i+1}^2) * (sin(50*(z_i^2 + z_{i+1}^2)^0.1) + 1)))^2
    特点：高度多峰函数，具有不规则的景观和许多局部最优解
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    f = 0.0
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    
    for i in range(nx - 1):
        # 注意：这里的y应该是z，可能是原代码的错误
        z[i] = pow(y[i] * y[i] + y[i + 1] * y[i + 1], 0.5)  # 计算相邻点间距离
        tmp = np.sin(50.0 * pow(z[i], 0.2))                  # 高频振荡项
        f += pow(z[i], 0.5) + pow(z[i], 0.5) * tmp * tmp     # Schaffer F7公式
    
    # 平方化并标准化
    f = f * f / (nx - 1) / (nx - 1)
    return f


def step_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    阶跃Rastrigin函数 (Non-Continuous Rastrigin Function)
    
    特点：在标准Rastrigin函数基础上添加阶跃特性，增加不连续性
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    f = 0.0
    
    # 阶跃处理：当距离原点超过0.5时进行取整操作
    for i in range(nx):
        if (np.fabs(y[i] - Os[i]) > 0.5):
            y[i] = Os[i] + np.floor(2 * (y[i] - Os[i]) + 0.5) / 2
    
    # 应用标准的sr变换
    z = sr_func(x, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag)
    
    # 计算Rastrigin函数值
    for i in range(nx):
        f += (z[i] * z[i] - 10.0 * np.cos(2.0 * PI * z[i]) + 10.0)
    return f


def levy_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Levy函数 (Levy Function)
    
    数学表达式: f(x) = sin^2(πw_1) + Σ((w_i-1)^2*(1+10*sin^2(πw_i+1))) + (w_n-1)^2*(1+sin^2(2πw_n))
    其中 w_i = 1 + (z_i - 1)/4
    特点：多峰函数，具有许多局部最优解
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    f = 0.0
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    
    # 初始化w数组：w_i = 1 + (z_i - 1)/4
    w = [1] * nx
    
    sum1 = 0.0
    for i in range(nx):
        w[i] = 1 + (z[i] - 1.0) / 4.0
    
    # 第一项：sin^2(πw_1)
    f = pow(np.sin(PI * w[0]), 2.0)
    
    # 中间项：Σ((w_i-1)^2*(1+10*sin^2(πw_i+1)))
    for i in range(nx - 1):
        f += pow(w[i] - 1, 2.0) * (1 + 10 * pow(np.sin(PI * w[i] + 1), 2.0))
    
    # 最后一项：(w_n-1)^2*(1+sin^2(2πw_n))
    f += pow(w[nx - 1] - 1, 2.0) * (1 + pow(np.sin(2 * PI * w[nx - 1]), 2.0))
    return f


def zakharov_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Zakharov函数 (Zakharov Function)
    
    数学表达式: f(x) = Σ(z_i^2) + (Σ(0.5*i*z_i))^2 + (Σ(0.5*i*z_i))^4
    特点：单峰函数，但具有非分离特性，各维度相互影响
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    f = 0.0
    sum1 = 0.0  # 二次项和
    sum2 = 0.0  # 加权和
    
    for i in range(nx):
        xi = z[i]
        sum1 = sum1 + pow(xi, 2)                # Σ(z_i^2)
        sum2 = sum2 + 0.5 * (i + 1) * xi        # Σ(0.5*i*z_i)
    
    # Zakharov函数公式：二次项 + 四次项 + 八次项
    f = sum1 + pow(sum2, 2) + pow(sum2, 4)
    return f


def katsuura_func(x, nx, Os, Mr, s_flag, r_flag):
    """
    Katsuura函数 (Katsuura Function)
    
    特点：高度多峰函数，在任何精度下都具有无穷多个局部最优解
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    f = 1.0
    
    for i in range(nx):
        temp = 0.0
        # Katsuura函数的无穷级数近似（使用32项）
        for j in range(32):
            temp += np.fabs(pow(2.0, j + 1) * z[i] - np.round(pow(2.0, j + 1) * z[i])) / pow(2.0, j + 1)
        
        # 乘积形式
        f *= pow(1.0 + (i + 1) * temp, 10.0 / pow(nx, 1.2))
    
    f = f * 10.0 / nx / nx - 10.0 / nx / nx
    return f


def hf02(x, nx, Os, Mr, S, s_flag, r_flag):
    """
    混合函数2 (Hybrid Function 2)
    
    组成：Bent Cigar (40%) + HGBat (40%) + Rastrigin (20%)
    特点：结合不同函数的特性，创建复杂的优化景观
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        S: 维度打乱向量
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    cf_num = 3  # 子函数数量
    fit = [None] * 3
    G = [None] * 3      # 各部分起始索引
    G_nx = [None] * 3   # 各部分维度数
    Gp = [0.4, 0.4, 0.2]  # 各部分比例：40%, 40%, 20%
    
    # 计算各部分的维度分配
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = np.ceil(Gp[i] * nx)
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G_nx = np.int64(G_nx)
    
    # 计算各部分的起始位置
    G[0] = 0
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]
    
    # 应用变换并打乱维度
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    S = list(map(int, S))
    for i in range(nx):
        y[i] = z[S[i] - 1]  # 根据S重新排列维度
    
    # 计算各子函数值
    i = 0
    fit[i] = bent_cigar_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 1
    fit[i] = hgbat_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 2
    fit[i] = rastrigin_func(y[G[i]:nx], G_nx[i], Os, Mr, 0, 0)
    
    # 简单求和
    f = 0.0
    for i in range(cf_num):
        f += fit[i]
    return f


def hf10(x, nx, Os, Mr, S, s_flag, r_flag):
    """
    混合函数10 (Hybrid Function 10)
    
    组成：HGBat (10%) + Katsuura (20%) + Ackley (20%) + Rastrigin (20%) + Schwefel (10%) + Schaffer F7 (20%)
    特点：包含6个不同的子函数，形成高度复杂的优化景观
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        S: 维度打乱向量
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    cf_num = 6
    fit = [None] * 6
    G = [None] * 6
    G_nx = [None] * 6
    Gp = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]  # 各部分比例
    
    # 维度分配
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = np.ceil(Gp[i] * nx)
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G_nx = np.int64(G_nx)
    
    # 计算起始位置
    G[0] = 0
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]
    
    # 变换和打乱
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    S = list(map(int, S))
    for i in range(nx):
        y[i] = z[S[i] - 1]
    
    # 计算各子函数
    i = 0
    fit[i] = hgbat_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 1
    fit[i] = katsuura_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 2
    fit[i] = ackley_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 3
    fit[i] = rastrigin_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 4
    fit[i] = schwefel_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 5
    fit[i] = schaffer_F7_func(y[G[i]:nx], G_nx[i], Os, Mr, 0, 0)
    
    # 求和
    f = 0.0
    for i in range(cf_num):
        f += fit[i]
    return f


def hf06(x, nx, Os, Mr, S, s_flag, r_flag):
    """
    混合函数6 (Hybrid Function 6)
    
    组成：Katsuura (30%) + HappyCat (20%) + Grie-Rosen (20%) + Schwefel (10%) + Ackley (20%)
    特点：包含5个子函数，平衡了不同的优化挑战
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        S: 维度打乱向量
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    cf_num = 5
    fit = [None] * 5
    G = [None] * 5
    G_nx = [None] * 5
    Gp = [0.3, 0.2, 0.2, 0.1, 0.2]  # 各部分比例
    
    # 维度分配
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = np.ceil(Gp[i] * nx)
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G_nx = np.int64(G_nx)
    
    # 计算起始位置
    G[0] = 0
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]
    
    # 变换和打乱
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    S = list(map(int, S))
    for i in range(nx):
        y[i] = z[S[i] - 1]
    
    # 计算各子函数
    i = 0
    fit[i] = katsuura_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 1
    fit[i] = happycat_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 2
    fit[i] = grie_rosen_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 3
    fit[i] = schwefel_func(y[G[i]:G[i + 1]], G_nx[i], Os, Mr, 0, 0)
    i = 4
    fit[i] = ackley_func(y[G[i]:nx], G_nx[i], Os, Mr, 0, 0)
    
    # 求和
    f = 0.0
    for i in range(cf_num):
        f += fit[i]
    return f


def cf01(x, nx, Os, Mr, s_flag, r_flag):
    """
    组合函数1 (Composition Function 1)
    
    组成：5个子函数的加权组合
    - Rosenbrock (权重10000/1e4)
    - Ellips (权重10000/1e10) 
    - Bent Cigar (权重10000/1e30)
    - Discus (权重10000/1e10)
    - Ellips (无旋转, 权重10000/1e10)
    
    特点：通过加权组合形成复杂的多峰景观
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量 (包含所有子函数的位移)
        Mr: 旋转矩阵 (包含所有子函数的旋转)
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    cf_num = 5  # 子函数数量
    fit = [None] * 5
    delta = [10, 20, 30, 40, 50]      # 各子函数的尺度参数
    bias = [0, 200, 300, 100, 400]    # 各子函数的偏置值
    
    # 计算各子函数值并进行标准化
    i = 0
    fit[i] = rosenbrock_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1e+4    # 标准化
    
    i = 1
    fit[i] = ellips_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1e+10
    
    i = 2
    fit[i] = bent_cigar_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1e+30
    
    i = 3
    fit[i] = discus_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1e+10
    
    i = 4
    fit[i] = ellips_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, 0)  # 无旋转
    fit[i] = 10000 * fit[i] / 1e+10
    
    # 使用cf_cal进行加权组合
    f = cf_cal(x, nx, Os, delta, bias, fit, cf_num)
    return f


def cf02(x, nx, Os, Mr, s_flag, r_flag):
    """
    组合函数2 (Composition Function 2)
    
    组成：3个子函数的加权组合
    - Schwefel (无旋转)
    - Rastrigin 
    - HGBat
    
    特点：包含全局欺骗性的Schwefel函数和多峰的Rastrigin函数
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    cf_num = 3
    fit = [None] * 3
    delta = [20, 10, 10]       # 尺度参数
    bias = [0, 200, 100]       # 偏置值
    
    i = 0
    fit[i] = schwefel_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, 0)  # 无旋转
    i = 1
    fit[i] = rastrigin_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    i = 2
    fit[i] = hgbat_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    
    f = cf_cal(x, nx, Os, delta, bias, fit, cf_num)
    return f


def cf06(x, nx, Os, Mr, s_flag, r_flag):
    """
    组合函数6 (Composition Function 6)
    
    组成：5个子函数的加权组合
    - Expanded Schaffer F6 (权重10000/2e7)
    - Schwefel
    - Griewank (权重1000/100)
    - Rosenbrock
    - Rastrigin (权重10000/1e3)
    
    特点：结合多种不同特性的函数，形成复杂景观
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    cf_num = 5
    fit = [None] * 5
    delta = [20, 20, 30, 30, 20]      # 尺度参数
    bias = [0, 200, 300, 400, 200]    # 偏置值
    
    i = 0
    fit[i] = escaffer6_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 2e+7    # 标准化
    
    i = 1
    fit[i] = schwefel_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    
    i = 2
    fit[i] = griewank_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 1000 * fit[i] / 100
    
    i = 3
    fit[i] = rosenbrock_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    
    i = 4
    fit[i] = rastrigin_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1e+3
    
    f = cf_cal(x, nx, Os, delta, bias, fit, cf_num)
    return f


def cf07(x, nx, Os, Mr, s_flag, r_flag):
    """
    组合函数7 (Composition Function 7)
    
    组成：6个子函数的加权组合
    - HGBat (权重10000/1000)
    - Rastrigin (权重10000/1e3)
    - Schwefel (权重10000/4e3)
    - Bent Cigar (权重10000/1e30)
    - Ellips (权重10000/1e10)
    - Expanded Schaffer F6 (权重10000/2e7)
    
    特点：最复杂的组合函数，包含6个不同特性的子函数
    
    参数:
        x: 输入向量
        nx: 问题维度
        Os: 位移向量
        Mr: 旋转矩阵
        s_flag: 位移标志
        r_flag: 旋转标志
    
    返回:
        f: 函数值
    """
    cf_num = 6
    fit = [None] * 6
    delta = [10, 20, 30, 40, 50, 60]           # 尺度参数
    bias = [0, 300, 500, 100, 400, 200]        # 偏置值
    
    i = 0
    fit[i] = hgbat_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1000
    
    i = 1
    fit[i] = rastrigin_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1e+3
    
    i = 2
    fit[i] = schwefel_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 4e+3
    
    i = 3
    fit[i] = bent_cigar_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1e+30
    
    i = 4
    fit[i] = ellips_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 1e+10
    
    i = 5
    fit[i] = escaffer6_func(x, nx, Os[i * nx:(i + 1) * nx], Mr[i * nx:(i + 1) * nx, 0:nx], 1, r_flag)
    fit[i] = 10000 * fit[i] / 2e+7
    
    f = cf_cal(x, nx, Os, delta, bias, fit, cf_num)
    return f


def shiftfunc(x, nx, Os):
    """
    位移变换函数 (Shift Transformation)
    
    将输入向量x根据位移向量Os进行位移
    
    参数:
        x: 输入向量 (会被就地修改)
        nx: 向量维度
        Os: 位移向量
    """
    for i in range(nx):
        x[i] = x[i] - Os[i]  # 将x向量向Os方向位移


def rotatefunc(x, nx, Mr):
    """
    旋转变换函数 (Rotation Transformation)
    
    使用旋转矩阵Mr对向量x进行旋转变换
    
    参数:
        x: 输入向量 (会被就地修改)
        nx: 向量维度
        Mr: 旋转矩阵 (nx × nx)
    """
    xrot = [0] * nx
    
    # 矩阵乘法：xrot = Mr × x
    for i in range(nx):
        xrot[i] = 0
        for j in range(nx):
            xrot[i] = xrot[i] + Mr[i][j] * x[j]
    
    # 将结果复制回原向量
    for i in range(nx):
        x[i] = xrot[i]


def sr_func(x, nx, Os, Mr, sh_rate, s_flag, r_flag):
    """
    位移-旋转-缩放组合变换函数 (Shift-Rotate-Scale Transformation)
    
    按顺序执行位移、旋转和缩放变换，这是CEC函数的标准预处理流程
    
    参数:
        x: 输入向量
        nx: 向量维度
        Os: 位移向量
        Mr: 旋转矩阵
        sh_rate: 缩放比例
        s_flag: 位移标志 (1=执行位移, 0=跳过)
        r_flag: 旋转标志 (1=执行旋转, 0=跳过)
    
    返回:
        y: 变换后的向量
    """
    y = [0] * nx
    
    # 复制输入向量
    for i in range(nx):
        y[i] = x[i]
    
    # 第1步：位移变换 (如果s_flag=1)
    if s_flag == 1:
        shiftfunc(y, nx, Os)
    
    # 第2步：旋转变换 (如果r_flag=1)  
    if r_flag == 1:
        rotatefunc(y, nx, Mr)
    
    # 第3步：缩放变换
    for i in range(nx):
        y[i] = y[i] * sh_rate
    
    return y


def asyfunc(x, xasy, nx, beta):
    """
    非对称变换函数 (Asymmetric Transformation)
    
    对正值施加非对称变换，使函数在某些方向上变得更难优化
    
    参数:
        x: 输入向量
        xasy: 输出向量
        nx: 向量维度
        beta: 非对称参数 (通常为0.5)
    """
    for i in range(nx):
        xasy[i] = x[i]
        # 只对正值进行非对称变换
        if (x[i] > 0):
            xasy[i] = pow(x[i], 1.0 + beta * i / (nx - 1) * pow(x[i], 0.5))


def oszfunc(x, xosz, nx):
    """
    振荡变换函数 (Oscillation Transformation)
    
    在第一维和最后一维施加对数-三角函数变换，增加函数的复杂性
    
    参数:
        x: 输入向量
        xosz: 输出向量
        nx: 向量维度
    """
    for i in range(nx):
        # 只对第一维和最后一维进行特殊处理
        if (i == 0 | i == nx - 1):
            if (x[i] != 0):
                xx = np.log(np.fabs(x[i]))  # 对数变换
                
            # 根据x[i]的符号选择不同参数
            if (x[i] > 0):
                c1 = 10
                c2 = 7.9
            else:
                c1 = 5.5
                c2 = 3.1
            
            # 符号函数
            if (x[i] > 0):
                sx = 1
            elif (x[i] == 0):
                sx = 0
            else:
                sx = -1
            
            # 振荡变换：指数函数结合正弦振荡
            xosz[i] = sx * np.exp(xx + 0.049 * (np.sin(c1 * xx) + np.sin(c2 * xx)))
        else:
            # 其他维度保持不变
            xosz[i] = x[i]


def cf_cal(x, nx, Os, delta, bias, fit, cf_num):
    """
    组合函数计算器 (Composition Function Calculator)
    
    使用加权平均方法组合多个子函数，权重基于距离各子函数最优解的远近
    
    参数:
        x: 当前点
        nx: 问题维度
        Os: 所有子函数的最优解位置 (展开为一维数组)
        delta: 各子函数的尺度参数
        bias: 各子函数的偏置值
        fit: 各子函数的函数值
        cf_num: 子函数数量
    
    返回:
        f: 组合后的函数值
        
    算法原理:
        1. 计算当前点到每个子函数最优解的距离
        2. 根据距离计算权重 (距离越近权重越大)
        3. 使用权重对各子函数值进行加权平均
    """
    w_max = 0      # 最大权重
    w_sum = 0      # 权重总和
    w = [None] * cf_num  # 权重数组
    
    for i in range(cf_num):
        fit[i] += bias[i]  # 添加偏置
        w[i] = 0
        
        # 计算到第i个子函数最优解的欧几里得距离的平方
        for j in range(nx):
            w[i] += pow(x[j] - Os[i * nx + j], 2.0)
        
        # 计算权重：距离越近权重越大
        if (w[i] != 0):
            w[i] = pow(1.0 / w[i], 0.5) * np.exp(-w[i] / 2.0 / nx / pow(delta[i], 2.0))
        else:
            w[i] = INF  # 如果距离为0，给予无穷大权重
            
        # 记录最大权重
        if (w[i] > w_max):
            w_max = w[i]
    
    # 计算权重总和
    for i in range(cf_num):
        w_sum = w_sum + w[i]
        
    # 如果所有权重都是0，则设置为均等权重
    if (w_max == 0):
        for i in range(cf_num):
            w[i] = 1
        w_sum = cf_num
    
    # 加权平均计算最终函数值
    f = 0.0
    for i in range(cf_num):
        f = f + w[i] / w_sum * fit[i]
    
    del (w)
    return f


def cec22_test_func(x, nx, mx, func_num):
    global OShift, M, y, z, x_bound, ini_flag, n_flag, func_flag, SS

    OShift = None
    M = None
    y = None
    z = None
    x_bound = None
    ini_flag = 0
    n_flag = None
    func_flag = None
    SS = None
    cf_num = 10
    if (func_num < 1) | (func_num > 12):
        print('\nError: Test function %d is not defined.\n' % func_num)
    if ini_flag == 1:
        if (n_flag != nx) | (func_flag != func_num):
            ini_flag = 0

    if ini_flag == 0:
        del (M)
        del (OShift)
        del (y)
        del (z)
        del (x_bound)
        y = [0] * nx
        z = [None] * nx
        x_bound = [100.0] * nx

        if (nx != 2 | nx != 10 | nx != 20):
            print("\nError: Test functions are only defined for D=2,10,20.\n")

        if (nx == 2) & (func_num == 6 | func_num == 7 | func_num == 8):
            print("\nError:  NOT defined for D=2.\n")

        # Load M matrix
        FileName = 'input_data/M_%d_D%d.txt' % (func_num, nx)
        try:
            M = np.loadtxt(FileName)
        except:
            print("\n Error: Cannot open M_%d_D%d.txt for reading \n" % (func_num, nx))
        del (FileName)

        # Shift data
        FileName = "input_data/shift_data_%d.txt" % func_num
        try:
            OShift_temp = np.loadtxt(FileName)
        except:
            print("\n Error: Cannot open shift_data_%d.txt for reading \n" % func_num)
        del (FileName)
        if (func_num < 9):
            OShift = np.zeros((nx,))
            for i in range(nx):
                OShift[i] = OShift_temp[i]
        else:
            OShift = np.zeros((cf_num - 1, nx))
            for i in range(cf_num - 1):
                for j in range(nx):
                    OShift[i, j] = OShift_temp[i, j]
            OShift = np.reshape(OShift, (cf_num - 1) * nx)
        if (func_num >= 6) & (func_num <= 8):
            FileName = "input_data/shuffle_data_%d_D%d.txt" % (func_num, nx)
            try:
                SS = np.loadtxt(FileName)
            except:
                print("\n Error: Cannot open shuffle_data_%d_D%d.txt for reading \n" % (func_num, nx))

            del (FileName)

        n_flag = nx
        func_flag = func_num
        ini_flag = 1
    f = np.zeros((mx,))
    for i in range(mx):
        if func_num == 1:
            ff = zakharov_func(x, nx, OShift, M, 1, 1)
            f[i] = ff + 300.0
            break
        elif func_num == 2:
            ff = rosenbrock_func(x, nx, OShift, M, 1, 1)
            f[i] = ff + 400.0
            break
        elif func_num == 3:
            ff = schaffer_F7_func(x, nx, OShift, M, 1, 1)
            f[i] = ff + 600.0
            break
        elif func_num == 4:
            ff = step_rastrigin_func(x, nx, OShift, M, 1, 1)
            f[i] = ff + 800.0
            break
        elif func_num == 5:
            ff = levy_func(x, nx, OShift, M, 1, 1)
            f[i] = ff + 900.0
            break
        elif func_num == 6:
            ff = hf02(x, nx, OShift, M, SS, 1, 1)
            f[i] = ff + 1800.0
            break
        elif func_num == 7:
            ff = hf10(x, nx, OShift, M, SS, 1, 1)
            f[i] = ff + 2000.0
            break
        elif func_num == 8:
            ff = hf06(x, nx, OShift, M, SS, 1, 1)
            f[i] = ff + 2200.0
            break
        elif func_num == 9:
            ff = cf01(x, nx, OShift, M, 1, 1)
            f[i] = ff + 2300.0
            break
        elif func_num == 10:
            ff = cf02(x, nx, OShift, M, 1, 1)
            f[i] = ff + 2400.0
            break
        elif func_num == 11:
            ff = cf06(x, nx, OShift, M, 1, 1)
            f[i] = ff + 2600.0
            break
        elif func_num == 12:
            ff = cf07(x, nx, OShift, M, 1, 1)
            f[i] = ff + 2700.0
            break
        else:
            print("\nError: There are only 10 test functions in this test suite!\n")
            f[i] = 0.0
            break

    return f


class cec2022_func():

    def __init__(self, func_num):
        self.func = func_num

    def values(self, x):
        nx, mx = np.shape(x)
        ObjFunc = np.zeros((mx,))
        for i in range(mx):
            ObjFunc[i] = cec22_test_func(x[:, i], nx, 1, self.func)
        self.ObjFunc = ObjFunc

        return self