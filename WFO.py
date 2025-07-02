# WFO: Water Flow Optimizer
# Reference: 
# Kaiping Luo. Water Flow Optimizer: a nature-inspired evolutionary algorithm for global optimization.
# IEEE Transactions on Cybernetics, 2021.
# ------------------------------------------------------
# Syntax   WFO.WFO(alg, prob)
#
# alg is a class, which includes the following fields:
# alg.NP:  the number of water particle
# alg.max_nfe: the maximal number of function evaluation
# alg.pl: the laminar probability
# al.pe: the eddying probability
#
# prob is also a class, which includes the following fields:
# prob.lb:  a row vector representing the lower bound
# prob.ub:  a row vector representing the upper bound
# prob.fobj:  a function representing the objective to be minimized
# prob.dim:  the dimension of the given problem
#
# The WFO function will return three arguments:
# fb: the best objective function value found
# xb: the best solution found
# con: convergence
# --------------------------------------------------------
# Edited by: Kaiping Luo, Beihang University, kaipingluo@buaa.edu.cn
# in Python 3

# 导入必要的数学和随机数库
from numpy import zeros,transpose  # numpy库用于矩阵操作
from math import inf,pi,cos  # 数学函数：无穷大、圆周率、余弦函数
from random import random,uniform,randrange  # 随机数生成函数

def WFO(alg, prob):
    """
    水流优化算法主函数
    参数:
        alg: 算法参数对象，包含粒子数、最大函数评估次数、层流概率、涡流概率
        prob: 问题参数对象，包含边界、目标函数、维度信息
    返回:
        fb: 找到的最优目标函数值
        xb: 找到的最优解
        con: 收敛曲线
    """
    global fb, xb, con
    fb = inf  # 初始化最优函数值为无穷大
    con = zeros((alg.max_nfe,1))  # 创建收敛曲线记录数组
    X = zeros((alg.NP, prob.dim))  # 创建水粒子位置矩阵 (粒子数 × 维度)
    F = zeros((alg.NP, 1))  # 创建水粒子适应度值数组
    
    
    # 粒子群初始化阶段
    for i in range(alg.NP): # initialization
        # 为每个水粒子在搜索空间内随机初始化位置
        for j in range(prob.dim):
            X[i,j] = uniform(prob.lb[j],prob.ub[j])  # 在第j维的上下界之间随机生成位置
        
        # 计算当前粒子的目标函数值
        F[i] = prob.fobj(X[i,:])
        
        # 更新全局最优解
        if F[i]<fb:
            fb = F[i]  # 更新最优函数值
            xb = X[i,:]  # 更新最优解位置
        
        con[i] = fb  # 记录当前的最优值到收敛曲线
    
    Y = zeros((alg.NP, prob.dim))  # 创建新位置矩阵，用于存储更新后的粒子位置
    nfe = i  # 记录当前的函数评估次数
    
    # 主优化循环 - 模拟水流运动
    while nfe < alg.max_nfe-1:
        # 根据层流概率决定采用层流还是湍流模式
        if random() < alg.pl: # laminar flow 层流模式
            # 层流模式：粒子向最优位置靠近
            k = randrange(alg.NP)  # 随机选择一个粒子作为参考
            d = xb - X[k,:]  # 计算最优位置与参考粒子之间的方向向量
            
            # 更新所有粒子位置
            for i in range(alg.NP):
                Y[i,:] = X[i,:] + random()*d  # 沿着最优方向移动，移动距离随机
                
                # 边界处理：确保新位置在可行域内
                for j in range(prob.dim):
                    if not prob.lb[j] <= Y[i,j] <= prob.ub[j]:
                        Y[i,j] = X[i,j]  # 如果超出边界，保持原位置
                        
        else: # turbulent flow 湍流模式
            # 湍流模式：粒子进行更复杂的运动
            for i in range(alg.NP):
                Y[i,:] = X[i,:]  # 先复制当前位置
                
                # 随机选择一个不同的粒子进行交互
                k = randrange(alg.NP)
                while k==i:  # 确保选择的粒子不是自己
                    k = randrange(alg.NP)
                
                j1 = randrange(prob.dim)  # 随机选择一个维度进行更新
                
                # 根据涡流概率决定运动模式
                if random() < alg.pe: # spiral flow 螺旋流模式
                    # 螺旋流：模拟水的螺旋运动
                    theta = uniform(-pi, pi)  # 随机角度
                    Y[i,j1] = X[i,j1]+abs(X[k,j1]-X[i,j1])*theta*cos(theta)  # 螺旋运动公式
                    
                    # 边界检查
                    if not prob.lb[j1] <= Y[i,j1] <= prob.ub[j1]:
                        Y[i,j1] = X[i,j1]  # 超出边界则保持原位置
                else:
                    # 交叉运动：基于其他粒子的位置信息进行维度交换
                    j2 = randrange(prob.dim)  # 选择另一个维度
                    while j2==j1:  # 确保两个维度不同
                        j2 = randrange(prob.dim)
                    
                    # 基于参考粒子在j2维的相对位置来更新当前粒子在j1维的位置
                    Y[i,j1] = prob.lb[j1] + (prob.ub[j1]-prob.lb[j1])*(X[k,j2]-prob.lb[j2])/(prob.ub[j2]-prob.lb[j2])
                    
        # 评估新位置并更新粒子群
        for i in range(alg.NP): # evaluation and evolution
            f = prob.fobj(Y[i,:])  # 计算新位置的目标函数值
            
            # 贪婪选择：如果新位置更好则接受
            if f < F[i]:
                F[i] = f  # 更新适应度值
                X[i,:] = Y[i,:]  # 更新粒子位置
                
                # 检查是否找到了新的全局最优解
                if f < fb:
                    fb = f  # 更新全局最优函数值
                    xb = X[i,:]  # 更新全局最优位置
            
            nfe += 1  # 增加函数评估次数计数
            con[nfe] = fb  # 记录当前最优值到收敛曲线
    
    return fb, xb,con  # 返回最优函数值、最优解和收敛曲线
