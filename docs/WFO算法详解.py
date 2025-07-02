"""
WFO水流优化算法详细机制解析
============================

本文档详细解析WFO (Water Flow Optimizer) 算法的工作原理、
参数机制、流程控制、数学模型等核心内容。

参考文献：
Kaiping Luo. Water Flow Optimizer: a nature-inspired evolutionary algorithm 
for global optimization. IEEE Transactions on Cybernetics, 2021.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp
from random import random, uniform, randrange

class WFOAnalyzer:
    """
    WFO算法分析器 - 用于深入理解算法机制
    """
    
    def __init__(self):
        self.flow_types = ['层流 (Laminar Flow)', '湍流 (Turbulent Flow)', '螺旋流 (Spiral Flow)']
        self.parameters = {
            'NP': '水粒子数量 (种群大小)',
            'max_nfe': '最大函数评估次数', 
            'pl': '层流概率 (0.6-0.8推荐)',
            'pe': '涡流概率 (0.2-0.4推荐)'
        }
    
    def explain_algorithm_principle(self):
        """解释算法基本原理"""
        
        principle = """
=== WFO算法基本原理 ===

1. 自然启发来源：
   - 水流在自然环境中的运动规律
   - 层流：有序、平稳的流动状态
   - 湍流：复杂、不规则的流动状态
   - 螺旋流：涡旋运动模式

2. 算法类比：
   - 水粒子 ↔ 搜索个体 
   - 水流运动 ↔ 搜索策略
   - 流动方向 ↔ 搜索方向
   - 能量转换 ↔ 解的更新

3. 核心机制：
   - 层流模式：全局探索，向最优解靠近
   - 湍流模式：局部开发，精细搜索
   - 概率控制：平衡探索与开发

4. 算法优势：
   - 参数少：只需4个主要参数
   - 收敛快：有效的方向引导
   - 鲁棒性强：适用于多种问题
   - 实现简单：代码易于理解和修改
"""
        print(principle)
    
    def analyze_flow_patterns(self):
        """分析不同流动模式的数学模型"""
        
        flow_analysis = """
=== 流动模式数学模型 ===

1. 层流模式 (Laminar Flow):
   数学公式：Y[i] = X[i] + r * (X_best - X[k])
   
   其中：
   - Y[i]: 粒子i的新位置
   - X[i]: 粒子i的当前位置  
   - X_best: 当前全局最优位置
   - X[k]: 随机选择的参考粒子位置
   - r: [0,1]的随机数
   
   特点：
   - 所有粒子朝向全局最优解移动
   - 移动距离随机，增加探索性
   - 保持群体的聚集性

2. 湍流模式 (Turbulent Flow):
   包含两种子模式：
   
   a) 螺旋流 (Spiral Flow):
      公式：Y[i,j] = X[i,j] + |X[k,j] - X[i,j]| * θ * cos(θ)
      
      其中：
      - θ: [-π, π]的随机角度
      - j: 随机选择的维度
      - 模拟水流的螺旋运动
   
   b) 交叉流 (Crossover Flow):
      公式：Y[i,j1] = lb[j1] + (ub[j1]-lb[j1]) * (X[k,j2]-lb[j2])/(ub[j2]-lb[j2])
      
      其中：
      - j1, j2: 不同的维度索引
      - 实现维度间的信息交换

3. 概率控制机制：
   - P(层流) = pl (通常 0.6-0.8)
   - P(湍流) = 1 - pl
   - P(螺旋流|湍流) = pe (通常 0.2-0.4)
   - P(交叉流|湍流) = 1 - pe
"""
        print(flow_analysis)
    
    def demonstrate_search_behavior(self):
        """演示搜索行为的可视化"""
        
        def simple_2d_function(x, y):
            """简单的2D测试函数"""
            return (x-2)**2 + (y-1)**2
        
        # 模拟WFO在2D空间的搜索过程
        NP = 10  # 粒子数
        dim = 2  # 2维问题
        bounds = [(-5, 5), (-5, 5)]  # 搜索边界
        
        # 初始化粒子
        particles = np.random.uniform(-5, 5, (NP, dim))
        best_pos = np.array([2.0, 1.0])  # 真实最优解
        
        # 记录搜索轨迹
        trajectory = [particles.copy()]
        
        # 模拟几次迭代
        for iter in range(5):
            new_particles = particles.copy()
            
            # 层流模式示例
            if random() < 0.7:  # pl = 0.7
                k = randrange(NP)
                direction = best_pos - particles[k]
                for i in range(NP):
                    new_particles[i] = particles[i] + random() * direction
            else:
                # 湍流模式示例
                for i in range(NP):
                    k = randrange(NP)
                    while k == i:
                        k = randrange(NP)
                    
                    j = randrange(dim)
                    if random() < 0.3:  # pe = 0.3
                        # 螺旋流
                        theta = uniform(-pi, pi)
                        new_particles[i,j] = particles[i,j] + abs(particles[k,j] - particles[i,j]) * theta * cos(theta)
            
            # 边界处理
            new_particles = np.clip(new_particles, -5, 5)
            particles = new_particles
            trajectory.append(particles.copy())
        
        return trajectory
    
    def analyze_parameters(self):
        """详细分析算法参数的作用机制"""
        
        param_analysis = """
=== 参数详细分析 ===

1. 种群大小 (NP):
   - 作用：控制搜索的并行度
   - 取值范围：20-100 (根据问题复杂度)
   - 影响：
     * 过小：搜索能力不足，容易陷入局部最优
     * 过大：计算成本高，收敛缓慢
   - 推荐：维度 × 2 ~ 维度 × 5

2. 最大函数评估次数 (max_nfe):
   - 作用：控制算法终止条件
   - 取值：根据问题难度和时间约束
   - 计算方式：NP × 代数 
   - 推荐：维度 × 1000 ~ 维度 × 10000

3. 层流概率 (pl):
   - 作用：控制全局探索强度
   - 取值范围：0.5 - 0.9
   - 影响：
     * 过高：过度探索，收敛慢
     * 过低：容易早熟收敛
   - 动态调整策略：
     * 初期：高值 (0.8) - 强化探索
     * 后期：低值 (0.5) - 强化开发

4. 涡流概率 (pe):
   - 作用：控制湍流模式中螺旋流的比例
   - 取值范围：0.1 - 0.5
   - 影响：
     * 螺旋流：局部精细搜索
     * 交叉流：维度间信息交换
   - 推荐：0.2 - 0.4

5. 参数敏感性分析：
   - pl 和 pe 是最关键参数
   - pl 对收敛速度影响最大
   - pe 对搜索精度影响显著
   - 不同问题类型需要不同参数组合
"""
        print(param_analysis)
    
    def compare_with_other_algorithms(self):
        """与其他算法的比较分析"""
        
        comparison = """
=== 与其他算法的比较 ===

1. vs 粒子群优化 (PSO):
   相似点：
   - 都基于群体智能
   - 都有全局最优引导
   
   不同点：
   - WFO无需速度概念，更简单
   - WFO的湍流模式提供更好的局部搜索
   - WFO参数更少，调参更容易

2. vs 遗传算法 (GA):
   相似点：
   - 都是进化算法
   - 都有选择机制
   
   不同点：
   - WFO基于物理模型，更直观
   - WFO不需要交叉变异操作
   - WFO收敛速度通常更快

3. vs 人工蜂群 (ABC):
   相似点：
   - 都基于自然现象
   - 都有探索-开发平衡
   
   不同点：
   - WFO的流动模式更丰富
   - WFO的数学模型更简洁
   - WFO的实现更直接

4. WFO算法优势：
   - 概念清晰：基于直观的物理现象
   - 参数少：只需4个主要参数
   - 实现简单：代码结构清晰
   - 性能好：在多数问题上表现优秀
   - 泛化强：适用于各种优化问题
"""
        print(comparison)
    
    def suggest_improvements(self):
        """改进建议和扩展方向"""
        
        improvements = """
=== 算法改进建议 ===

1. 自适应参数调整：
   - 动态调整pl和pe值
   - 基于收敛状态自动调参
   - 实现策略：
     * pl = 0.9 - 0.4 * (current_iter / max_iter)
     * pe = 0.1 + 0.3 * (current_iter / max_iter)

2. 多种流动模式：
   - 增加更多流动类型
   - 瀑布流：大幅度跳跃搜索
   - 渗流：缓慢渐进搜索
   - 波浪流：周期性搜索

3. 精英保留机制：
   - 保护最优个体不被破坏
   - 精英引导其他个体
   - 多样性维护

4. 混合策略：
   - 与其他算法结合
   - 多阶段搜索策略
   - 局部搜索增强

5. 约束处理：
   - 惩罚函数法
   - 修复策略
   - 可行性规则

6. 多目标扩展：
   - Pareto支配关系
   - 非支配排序
   - 拥挤度距离

7. 并行实现：
   - 多线程并行评估
   - 分布式计算
   - GPU加速
"""
        print(improvements)

def create_wfo_variants():
    """创建WFO算法的改进变种"""
    
    variants_code = '''
# === WFO算法改进变种 ===

class AdaptiveWFO:
    """自适应WFO算法"""
    
    def __init__(self, alg, prob):
        self.alg = alg
        self.prob = prob
        self.pl_initial = alg.pl
        self.pe_initial = alg.pe
    
    def adaptive_parameters(self, current_iter, max_iter):
        """自适应参数调整"""
        progress = current_iter / max_iter
        
        # 线性递减层流概率
        self.alg.pl = self.pl_initial * (1 - 0.5 * progress)
        
        # 线性递增涡流概率  
        self.alg.pe = self.pe_initial * (1 + progress)
        
        return self.alg.pl, self.alg.pe

class EnhancedWFO:
    """增强WFO算法 - 添加新的流动模式"""
    
    def waterfall_flow(self, X, i, prob):
        """瀑布流模式 - 大幅度跳跃"""
        new_pos = X[i].copy()
        
        # 随机选择几个维度进行大幅跳跃
        num_jumps = max(1, prob.dim // 4)
        jump_dims = np.random.choice(prob.dim, num_jumps, replace=False)
        
        for dim in jump_dims:
            # 在整个搜索空间内随机跳跃
            new_pos[dim] = uniform(prob.lb[dim], prob.ub[dim])
        
        return new_pos
    
    def wave_flow(self, X, i, t, prob):
        """波浪流模式 - 周期性搜索"""
        new_pos = X[i].copy()
        
        for j in range(prob.dim):
            # 添加正弦波动
            amplitude = (prob.ub[j] - prob.lb[j]) * 0.1
            frequency = 2 * pi / 100  # 100步一个周期
            wave = amplitude * sin(frequency * t)
            
            new_pos[j] += wave
            # 边界处理
            new_pos[j] = max(prob.lb[j], min(prob.ub[j], new_pos[j]))
        
        return new_pos

class HybridWFO:
    """混合WFO算法 - 结合局部搜索"""
    
    def local_search(self, x, prob, radius=0.1):
        """简单的局部搜索"""
        best_x = x.copy()
        best_f = prob.fobj(x)
        
        # 在当前解附近进行小范围搜索
        for _ in range(10):
            new_x = x.copy()
            for j in range(prob.dim):
                range_size = (prob.ub[j] - prob.lb[j]) * radius
                new_x[j] += uniform(-range_size, range_size)
                new_x[j] = max(prob.lb[j], min(prob.ub[j], new_x[j]))
            
            new_f = prob.fobj(new_x)
            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()
        
        return best_x, best_f
'''
    
    print("=== WFO算法变种代码 ===")
    print(variants_code)

def performance_analysis():
    """WFO算法性能分析"""
    
    analysis = """
=== WFO算法性能分析 ===

1. 时间复杂度：
   - 每代时间复杂度：O(NP × D)
   - 总时间复杂度：O(max_nfe × D)
   - 与其他群智能算法相当

2. 空间复杂度：
   - 主要存储：O(NP × D) - 粒子位置
   - 辅助存储：O(NP) - 适应度值
   - 总空间复杂度：O(NP × D)

3. 收敛性分析：
   - 全局收敛性：理论上可证明
   - 收敛速度：通常较快
   - 稳定性：表现稳定

4. 适用问题类型：
   - 连续优化问题：★★★★★
   - 多峰函数：★★★★☆
   - 高维问题：★★★★☆  
   - 约束优化：★★★☆☆
   - 多目标优化：★★★☆☆ (需扩展)

5. 参数敏感性：
   - pl：敏感度高 ★★★★☆
   - pe：敏感度中 ★★★☆☆
   - NP：敏感度低 ★★☆☆☆
   - max_nfe：敏感度低 ★☆☆☆☆

6. 实际应用建议：
   - 工程优化：优先考虑
   - 机器学习：参数优化
   - 神经网络：权重训练
   - 信号处理：滤波器设计
   - 控制系统：参数整定
"""
    print(analysis)

if __name__ == "__main__":
    # 创建分析器实例
    analyzer = WFOAnalyzer()
    
    print("=== WFO水流优化算法详细机制解析 ===\n")
    
    # 1. 基本原理
    analyzer.explain_algorithm_principle()
    
    # 2. 流动模式分析
    analyzer.analyze_flow_patterns()
    
    # 3. 参数分析
    analyzer.analyze_parameters()
    
    # 4. 算法比较
    analyzer.compare_with_other_algorithms()
    
    # 5. 改进建议
    analyzer.suggest_improvements()
    
    # 6. 创建变种
    create_wfo_variants()
    
    # 7. 性能分析
    performance_analysis()
    
    print("\n=== 分析完成 ===")
    print("WFO算法是一个简单而有效的优化算法，")
    print("具有清晰的物理意义和良好的性能表现。") 