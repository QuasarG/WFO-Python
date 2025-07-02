"""
WFO项目扩展完整指南
====================

本指南详细说明如何扩展WFO项目，支持：
1. 更高维度的优化问题 (>20维)
2. 更多的测试函数
3. 新的算法变种
4. 性能优化和并行化
5. 用户界面增强
"""

import numpy as np
import os
import shutil
from pathlib import Path

class ProjectExtender:
    """
    项目扩展器 - 提供系统化的扩展方法
    """
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.supported_dimensions = [2, 10, 20]  # 当前支持的维度
        self.max_functions = 12  # 当前最大函数数量
    
    def extend_dimension_support(self):
        """扩展维度支持指南"""
        
        guide = """
=== 扩展高维度支持 (>20维) ===

当前限制：
- CEC2022函数仅支持2, 10, 20维
- 数据文件只包含这些维度的旋转矩阵和位移向量
- GUI界面有维度限制提示

扩展步骤：

1. 修改CEC2022.py中的维度检查：
```python
# 当前代码：
if (nx != 2 | nx != 10 | nx != 20):
    print("Error: Test functions are only defined for D=2,10,20.")

# 修改为：
SUPPORTED_DIMS = [2, 10, 20, 30, 50, 100]  # 添加新维度
if nx not in SUPPORTED_DIMS:
    print(f"Error: Supported dimensions: {SUPPORTED_DIMS}")
```

2. 生成新维度的数据文件：
```python
def generate_high_dim_data(func_num, new_dimensions):
    \"\"\"为新维度生成数据文件\"\"\"
    
    for dim in new_dimensions:
        # 生成旋转矩阵
        if func_num <= 8:  # 需要旋转矩阵的函数
            rotation_matrix = generate_rotation_matrix(dim)
            np.savetxt(f'input_data/M_{func_num}_D{dim}.txt', rotation_matrix)
        
        # 生成位移向量
        if func_num < 9:  # 单函数
            shift_vector = np.random.uniform(-80, 80, dim)
        else:  # 组合函数
            cf_num = get_composition_number(func_num)
            shift_vector = np.random.uniform(-80, 80, (cf_num-1, dim))
        
        np.savetxt(f'input_data/shift_data_{func_num}_D{dim}.txt', shift_vector)
        
        # 生成打乱向量 (仅混合函数需要)
        if 6 <= func_num <= 8:
            shuffle_vector = np.random.permutation(dim) + 1  # 1-based indexing
            np.savetxt(f'input_data/shuffle_data_{func_num}_D{dim}.txt', shuffle_vector)

def generate_rotation_matrix(dim):
    \"\"\"生成正交随机旋转矩阵\"\"\"
    # 方法1：QR分解
    A = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    
    # 确保det(Q) = 1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    
    return Q

def generate_rotation_matrix_householder(dim):
    \"\"\"使用Householder反射生成旋转矩阵 - 更稳定\"\"\"
    Q = np.eye(dim)
    
    for i in range(dim - 1):
        # 生成随机向量
        v = np.random.randn(dim - i)
        v = v / np.linalg.norm(v)
        
        # Householder反射矩阵
        H = np.eye(dim - i) - 2 * np.outer(v, v)
        
        # 嵌入到大矩阵中
        Q_temp = np.eye(dim)
        Q_temp[i:, i:] = H
        Q = Q @ Q_temp
    
    return Q
```

3. 修改GUI界面支持新维度：
```python
# 在WFO_exe.py中修改维度输入验证
def validate_dimension(dim_input):
    \"\"\"验证维度输入\"\"\"
    try:
        dim = int(dim_input)
        max_supported_dim = 100  # 提高维度上限
        
        if dim < 2:
            return False, "维度不能小于2"
        elif dim > max_supported_dim:
            return False, f"维度不能大于{max_supported_dim}"
        else:
            return True, "维度有效"
    except ValueError:
        return False, "请输入有效的整数"

# 修改提示信息
createToolTip(dimEntered, '支持2-100维优化问题')
```

4. 性能优化建议：
```python
class HighDimOptimizer:
    \"\"\"高维优化器 - 针对高维问题的优化\"\"\"
    
    def __init__(self):
        self.use_sparse_matrix = True  # 对于稀疏旋转矩阵
        self.batch_evaluation = True   # 批量函数评估
        self.dimension_reduction = True # 维度约简选项
    
    def adaptive_population_size(self, dimension):
        \"\"\"根据维度自适应调整种群大小\"\"\"
        if dimension <= 20:
            return max(20, dimension * 2)
        elif dimension <= 50:
            return max(30, dimension * 1.5)
        else:
            return max(50, dimension)
    
    def adaptive_max_nfe(self, dimension):
        \"\"\"根据维度自适应调整最大评估次数\"\"\"
        base_nfe = 10000
        if dimension <= 20:
            return base_nfe
        elif dimension <= 50:
            return base_nfe * 2
        else:
            return base_nfe * 3
```

5. 内存优化：
```python
def memory_efficient_wfo(alg, prob):
    \"\"\"内存高效的WFO实现\"\"\"
    
    # 使用生成器减少内存占用
    def particle_generator():
        for i in range(alg.NP):
            yield np.random.uniform(prob.lb, prob.ub, prob.dim)
    
    # 分批处理大种群
    batch_size = min(alg.NP, 100)
    
    for batch_start in range(0, alg.NP, batch_size):
        batch_end = min(batch_start + batch_size, alg.NP)
        # 处理当前批次...
```
"""
        print(guide)
    
    def add_new_test_functions(self):
        """添加新测试函数的指南"""
        
        guide = """
=== 添加新测试函数 ===

方法1：扩展CEC2022框架

1. 在CEC2022.py中添加新函数：
```python
def my_new_function(x, nx, Os, Mr, s_flag, r_flag):
    \"\"\"
    新的测试函数
    例如：Styblinski-Tang函数
    \"\"\"
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    
    f = 0.0
    for i in range(nx):
        f += z[i]**4 - 16*z[i]**2 + 5*z[i]
    
    return f / 2.0  # 标准化

def goldstein_price_nd(x, nx, Os, Mr, s_flag, r_flag):
    \"\"\"
    高维Goldstein-Price函数变种
    \"\"\"
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    
    f = 1.0
    for i in range(0, nx-1, 2):
        if i+1 < nx:
            x1, x2 = z[i], z[i+1]
            
            term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
            term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
            
            f *= (term1 * term2)
    
    return np.log(f)  # 取对数避免数值溢出

def dixon_price_func(x, nx, Os, Mr, s_flag, r_flag):
    \"\"\"
    Dixon-Price函数
    \"\"\"
    z = sr_func(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    
    f = (z[0] - 1)**2
    
    for i in range(1, nx):
        f += (i + 1) * (2*z[i]**2 - z[i-1])**2
    
    return f
```

2. 修改cec22_test_func添加新函数：
```python
def cec22_test_func(x, nx, mx, func_num):
    # ... 现有代码 ...
    
    for i in range(mx):
        # 添加新函数选项
        if func_num == 13:
            ff = my_new_function(x, nx, OShift, M, 1, 1)
            f[i] = ff + 3000.0
            break
        elif func_num == 14:
            ff = goldstein_price_nd(x, nx, OShift, M, 1, 1)
            f[i] = ff + 3100.0
            break
        elif func_num == 15:
            ff = dixon_price_func(x, nx, OShift, M, 1, 1)
            f[i] = ff + 3200.0
            break
        # ... 其他函数 ...
```

3. 生成新函数的数据文件：
```python
def generate_function_data():
    \"\"\"为新函数生成数据文件\"\"\"
    
    new_functions = [13, 14, 15]  # 新函数编号
    dimensions = [2, 10, 20, 30, 50]
    
    for func_num in new_functions:
        for dim in dimensions:
            # 生成位移数据
            shift_data = np.random.uniform(-80, 80, dim)
            np.savetxt(f'input_data/shift_data_{func_num}.txt', shift_data)
            
            # 生成旋转矩阵
            rotation_matrix = generate_rotation_matrix(dim)
            np.savetxt(f'input_data/M_{func_num}_D{dim}.txt', rotation_matrix)
```

方法2：创建独立的函数库

1. 创建新的函数模块：
```python
# new_functions.py
class NewFunctionSuite:
    \"\"\"新的测试函数套件\"\"\"
    
    def __init__(self):
        self.functions = {
            'styblinski_tang': self.styblinski_tang,
            'dixon_price': self.dixon_price,
            'schwefel_2_26': self.schwefel_2_26,
            'alpine': self.alpine,
            'zakharov_4': self.zakharov_4
        }
    
    def styblinski_tang(self, x):
        \"\"\"Styblinski-Tang函数\"\"\"
        return sum(xi**4 - 16*xi**2 + 5*xi for xi in x) / 2
    
    def dixon_price(self, x):
        \"\"\"Dixon-Price函数\"\"\"
        n = len(x)
        f = (x[0] - 1)**2
        for i in range(1, n):
            f += (i + 1) * (2*x[i]**2 - x[i-1])**2
        return f
    
    def schwefel_2_26(self, x):
        \"\"\"Schwefel 2.26函数\"\"\"
        n = len(x)
        return 418.9829 * n - sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)
    
    def alpine(self, x):
        \"\"\"Alpine函数\"\"\"
        return sum(abs(xi * np.sin(xi) + 0.1 * xi) for xi in x)
    
    def zakharov_4(self, x):
        \"\"\"4阶Zakharov函数\"\"\"
        n = len(x)
        sum1 = sum(xi**2 for xi in x)
        sum2 = sum(0.5 * (i+1) * xi for i, xi in enumerate(x))
        return sum1 + sum2**2 + sum2**4 + sum2**6  # 添加6次项
```

2. 集成到主框架：
```python
# 修改Demo.py使用新函数
from new_functions import NewFunctionSuite

new_suite = NewFunctionSuite()

class prob:
    dim = 10
    lb = [-5] * dim
    ub = [5] * dim
    
    def fobj(x):
        return new_suite.functions['styblinski_tang'](x)
```
"""
        print(guide)
    
    def create_algorithm_variants(self):
        """创建算法变种的指南"""
        
        guide = """
=== 创建WFO算法变种 ===

1. 多种群WFO (Multi-Swarm WFO):
```python
class MultiSwarmWFO:
    \"\"\"多种群WFO算法\"\"\"
    
    def __init__(self, alg, prob, num_swarms=3):
        self.num_swarms = num_swarms
        self.swarm_size = alg.NP // num_swarms
        self.alg = alg
        self.prob = prob
        
    def run(self):
        # 初始化多个种群
        swarms = []
        for i in range(self.num_swarms):
            swarm = self.initialize_swarm(self.swarm_size)
            swarms.append(swarm)
        
        # 独立进化各种群
        for iteration in range(self.alg.max_nfe // self.alg.NP):
            for swarm in swarms:
                self.evolve_swarm(swarm)
            
            # 种群间信息交换
            if iteration % 10 == 0:
                self.exchange_information(swarms)
        
        return self.get_best_solution(swarms)

2. 自适应WFO (Adaptive WFO):
```python
class AdaptiveWFO:
    \"\"\"自适应参数WFO算法\"\"\"
    
    def __init__(self, alg, prob):
        self.alg = alg
        self.prob = prob
        self.pl_history = []
        self.pe_history = []
        self.performance_history = []
    
    def adapt_parameters(self, iteration, performance):
        \"\"\"根据性能自适应调整参数\"\"\"
        
        # 记录历史
        self.performance_history.append(performance)
        
        if len(self.performance_history) > 10:
            # 计算改进率
            recent_improvement = (self.performance_history[-10] - 
                                self.performance_history[-1]) / self.performance_history[-10]
            
            if recent_improvement < 0.01:  # 改进缓慢
                self.alg.pl *= 0.95  # 减少层流概率
                self.alg.pe *= 1.05  # 增加涡流概率
            else:  # 改进良好
                self.alg.pl *= 1.02  # 轻微增加层流概率
                self.alg.pe *= 0.98  # 轻微减少涡流概率
        
        # 边界检查
        self.alg.pl = max(0.3, min(0.9, self.alg.pl))
        self.alg.pe = max(0.1, min(0.5, self.alg.pe))

3. 混合WFO (Hybrid WFO):
```python
class HybridWFO:
    \"\"\"混合局部搜索的WFO算法\"\"\"
    
    def __init__(self, alg, prob, local_search_prob=0.1):
        self.alg = alg
        self.prob = prob
        self.local_search_prob = local_search_prob
    
    def local_search(self, x):
        \"\"\"简单的局部搜索\"\"\"
        best_x = x.copy()
        best_f = self.prob.fobj(x)
        
        # 在当前解附近搜索
        for _ in range(5):
            new_x = x + np.random.normal(0, 0.1, len(x))
            # 边界处理
            new_x = np.clip(new_x, self.prob.lb, self.prob.ub)
            
            new_f = self.prob.fobj(new_x)
            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()
        
        return best_x, best_f
    
    def enhance_solution(self, x):
        \"\"\"解增强策略\"\"\"
        if np.random.random() < self.local_search_prob:
            return self.local_search(x)
        else:
            return x, self.prob.fobj(x)

4. 动态WFO (Dynamic WFO):
```python
class DynamicWFO:
    \"\"\"动态环境适应的WFO算法\"\"\"
    
    def __init__(self, alg, prob):
        self.alg = alg
        self.prob = prob
        self.change_detection = True
        self.diversity_threshold = 0.01
    
    def detect_environment_change(self, population, fitness):
        \"\"\"检测环境变化\"\"\"
        # 重新评估部分个体
        sample_size = max(5, len(population) // 10)
        sample_indices = np.random.choice(len(population), sample_size, replace=False)
        
        change_detected = False
        for i in sample_indices:
            new_fitness = self.prob.fobj(population[i])
            if abs(new_fitness - fitness[i]) > 0.01:
                change_detected = True
                break
        
        return change_detected
    
    def respond_to_change(self, population):
        \"\"\"响应环境变化\"\"\"
        # 重新初始化部分个体
        reinit_ratio = 0.3
        num_reinit = int(len(population) * reinit_ratio)
        
        for i in range(num_reinit):
            for j in range(self.prob.dim):
                population[i][j] = np.random.uniform(self.prob.lb[j], self.prob.ub[j])
```
"""
        print(guide)
    
    def performance_optimization(self):
        """性能优化指南"""
        
        guide = """
=== 性能优化策略 ===

1. 并行化实现：
```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class ParallelWFO:
    \"\"\"并行化WFO实现\"\"\"
    
    def __init__(self, alg, prob, n_processes=None):
        self.alg = alg
        self.prob = prob
        self.n_processes = n_processes or mp.cpu_count()
    
    def parallel_evaluation(self, population):
        \"\"\"并行评估种群\"\"\"
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            fitness_values = list(executor.map(self.prob.fobj, population))
        return fitness_values
    
    def parallel_evolution(self, population):
        \"\"\"并行进化\"\"\"
        chunks = np.array_split(population, self.n_processes)
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            futures = [executor.submit(self.evolve_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        
        return np.vstack(results)

2. 矢量化优化：
```python
class VectorizedWFO:
    \"\"\"矢量化WFO实现\"\"\"
    
    def vectorized_laminar_flow(self, X, xb):
        \"\"\"矢量化层流计算\"\"\"
        NP, dim = X.shape
        
        # 随机选择参考粒子
        k = np.random.randint(0, NP)
        
        # 矢量化计算方向
        direction = xb - X[k]
        
        # 矢量化更新所有粒子
        random_factors = np.random.random((NP, 1))
        Y = X + random_factors * direction
        
        return Y
    
    def vectorized_boundary_check(self, X, lb, ub):
        \"\"\"矢量化边界检查\"\"\"
        return np.clip(X, lb, ub)

3. 内存优化：
```python
class MemoryEfficientWFO:
    \"\"\"内存高效WFO实现\"\"\"
    
    def __init__(self, alg, prob):
        self.alg = alg
        self.prob = prob
        self.use_float32 = True  # 使用单精度浮点数
    
    def create_population(self):
        \"\"\"创建种群\"\"\"
        dtype = np.float32 if self.use_float32 else np.float64
        
        population = np.empty((self.alg.NP, self.prob.dim), dtype=dtype)
        
        for i in range(self.alg.NP):
            for j in range(self.prob.dim):
                population[i, j] = np.random.uniform(
                    self.prob.lb[j], self.prob.ub[j]
                ).astype(dtype)
        
        return population
    
    def in_place_update(self, X, Y, fitness_X, fitness_Y):
        \"\"\"原地更新避免额外内存分配\"\"\"
        improvement_mask = fitness_Y < fitness_X
        X[improvement_mask] = Y[improvement_mask]
        fitness_X[improvement_mask] = fitness_Y[improvement_mask]

4. 编译优化：
```python
# 使用Numba JIT编译
from numba import jit

@jit(nopython=True)
def jit_laminar_flow(X, xb, NP, dim):
    \"\"\"JIT编译的层流计算\"\"\"
    Y = np.empty_like(X)
    k = np.random.randint(0, NP)
    
    for i in range(NP):
        r = np.random.random()
        for j in range(dim):
            Y[i, j] = X[i, j] + r * (xb[j] - X[k, j])
    
    return Y

@jit(nopython=True)
def jit_spiral_flow(X, i, k, j, lb_j, ub_j):
    \"\"\"JIT编译的螺旋流计算\"\"\"
    theta = np.random.uniform(-np.pi, np.pi)
    new_value = X[i, j] + abs(X[k, j] - X[i, j]) * theta * np.cos(theta)
    
    if not (lb_j <= new_value <= ub_j):
        new_value = X[i, j]
    
    return new_value

5. GPU加速 (使用CuPy):
```python
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class GPUWFO:
    \"\"\"GPU加速WFO实现\"\"\"
    
    def __init__(self, alg, prob):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available for GPU computation")
        
        self.alg = alg
        self.prob = prob
    
    def gpu_evaluation(self, population_gpu):
        \"\"\"GPU上的函数评估\"\"\"
        # 将评估函数移植到GPU
        # 注意：需要函数支持GPU计算
        fitness_gpu = cp.zeros(population_gpu.shape[0])
        
        # 简单的二次函数示例
        for i in range(population_gpu.shape[0]):
            fitness_gpu[i] = cp.sum(population_gpu[i] ** 2)
        
        return fitness_gpu
```
"""
        print(guide)
    
    def gui_enhancement(self):
        """GUI增强指南"""
        
        guide = """
=== GUI界面增强 ===

1. 添加高级参数设置：
```python
# 在WFO_exe.py中添加高级设置窗口
def open_advanced_settings():
    \"\"\"打开高级设置窗口\"\"\"
    advanced_window = tk.Toplevel(top)
    advanced_window.title("高级设置")
    advanced_window.geometry("400x300")
    
    # 并行设置
    ttk.Label(advanced_window, text="并行进程数:").grid(row=0, column=0, sticky='W')
    process_var = tk.IntVar(value=4)
    process_spinbox = tk.Spinbox(advanced_window, from_=1, to=16, textvariable=process_var)
    process_spinbox.grid(row=0, column=1)
    
    # 精度设置
    ttk.Label(advanced_window, text="计算精度:").grid(row=1, column=0, sticky='W')
    precision_var = tk.StringVar(value="float64")
    precision_combo = ttk.Combobox(advanced_window, textvariable=precision_var, 
                                  values=["float32", "float64"])
    precision_combo.grid(row=1, column=1)
    
    # 算法变种选择
    ttk.Label(advanced_window, text="算法变种:").grid(row=2, column=0, sticky='W')
    variant_var = tk.StringVar(value="标准WFO")
    variant_combo = ttk.Combobox(advanced_window, textvariable=variant_var,
                                values=["标准WFO", "自适应WFO", "混合WFO", "多种群WFO"])
    variant_combo.grid(row=2, column=1)

2. 实时结果可视化：
```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class RealTimeVisualizer:
    \"\"\"实时结果可视化器\"\"\"
    
    def __init__(self, parent_frame):
        self.fig = Figure(figsize=(8, 6))
        self.ax1 = self.fig.add_subplot(211)  # 收敛曲线
        self.ax2 = self.fig.add_subplot(212)  # 多样性曲线
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.convergence_data = []
        self.diversity_data = []
    
    def update_plots(self, iteration, best_fitness, diversity):
        \"\"\"更新图表\"\"\"
        self.convergence_data.append(best_fitness)
        self.diversity_data.append(diversity)
        
        # 更新收敛曲线
        self.ax1.clear()
        self.ax1.plot(self.convergence_data)
        self.ax1.set_title("收敛曲线")
        self.ax1.set_ylabel("最优函数值")
        
        # 更新多样性曲线
        self.ax2.clear()
        self.ax2.plot(self.diversity_data, 'r-')
        self.ax2.set_title("种群多样性")
        self.ax2.set_ylabel("多样性指标")
        self.ax2.set_xlabel("迭代次数")
        
        self.canvas.draw()

3. 结果分析面板：
```python
class ResultAnalyzer:
    \"\"\"结果分析器\"\"\"
    
    def __init__(self, parent_frame):
        self.frame = ttk.LabelFrame(parent_frame, text="结果分析")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 统计信息显示
        self.create_statistics_panel()
        
        # 参数敏感性分析
        self.create_sensitivity_panel()
    
    def create_statistics_panel(self):
        \"\"\"创建统计信息面板\"\"\"
        stats_frame = ttk.LabelFrame(self.frame, text="运行统计")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 显示各种统计指标
        ttk.Label(stats_frame, text="收敛代数:").grid(row=0, column=0, sticky='W')
        self.convergence_gen_label = ttk.Label(stats_frame, text="--")
        self.convergence_gen_label.grid(row=0, column=1, sticky='W')
        
        ttk.Label(stats_frame, text="函数调用次数:").grid(row=1, column=0, sticky='W')
        self.function_calls_label = ttk.Label(stats_frame, text="--")
        self.function_calls_label.grid(row=1, column=1, sticky='W')
        
        ttk.Label(stats_frame, text="运行时间:").grid(row=2, column=0, sticky='W')
        self.runtime_label = ttk.Label(stats_frame, text="--")
        self.runtime_label.grid(row=2, column=1, sticky='W')

4. 参数优化助手：
```python
class ParameterTuner:
    \"\"\"参数调优助手\"\"\"
    
    def __init__(self, parent_frame):
        self.frame = ttk.LabelFrame(parent_frame, text="参数调优")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_auto_tune_panel()
    
    def create_auto_tune_panel(self):
        \"\"\"创建自动调优面板\"\"\"
        # 自动调优按钮
        auto_tune_btn = ttk.Button(self.frame, text="自动调优参数", 
                                  command=self.auto_tune_parameters)
        auto_tune_btn.pack(pady=5)
        
        # 调优进度条
        self.tune_progress = ttk.Progressbar(self.frame, mode='determinate')
        self.tune_progress.pack(fill=tk.X, padx=5, pady=5)
        
        # 推荐参数显示
        self.recommendation_text = tk.Text(self.frame, height=5, width=50)
        self.recommendation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def auto_tune_parameters(self):
        \"\"\"自动参数调优\"\"\"
        # 实现参数调优逻辑
        recommended_params = self.grid_search_parameters()
        
        recommendation = f'''
推荐参数设置：
NP (种群大小): {recommended_params['NP']}
pl (层流概率): {recommended_params['pl']:.2f}
pe (涡流概率): {recommended_params['pe']:.2f}
max_nfe (最大评估次数): {recommended_params['max_nfe']}

调优基于: 网格搜索 + 10次独立运行
平均性能提升: {recommended_params['improvement']:.2%}
'''
        
        self.recommendation_text.delete(1.0, tk.END)
        self.recommendation_text.insert(1.0, recommendation)
```
"""
        print(guide)

def create_extension_toolkit():
    """创建扩展工具包"""
    
    toolkit_code = '''
# === 项目扩展工具包 ===

class ProjectExtensionToolkit:
    """项目扩展工具包 - 自动化扩展流程"""
    
    def __init__(self, project_path="."):
        self.project_path = Path(project_path)
        self.backup_path = self.project_path / "backups"
        self.extensions_path = self.project_path / "extensions"
    
    def backup_project(self):
        """备份当前项目"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"backup_{timestamp}"
        
        # 复制项目文件
        shutil.copytree(self.project_path, backup_dir, 
                       ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'backups'))
        
        print(f"项目已备份到: {backup_dir}")
    
    def add_dimension_support(self, new_dimensions):
        """自动添加新维度支持"""
        print(f"添加维度支持: {new_dimensions}")
        
        # 1. 生成数据文件
        self.generate_data_files(new_dimensions)
        
        # 2. 修改源代码
        self.update_dimension_checks(new_dimensions)
        
        # 3. 更新GUI
        self.update_gui_dimension_limits(new_dimensions)
        
        print("维度支持扩展完成!")
    
    def add_test_function(self, func_name, func_code, func_properties):
        """自动添加新测试函数"""
        print(f"添加测试函数: {func_name}")
        
        # 1. 在CEC2022.py中添加函数
        self.insert_function_code(func_name, func_code)
        
        # 2. 修改主调用函数
        self.update_function_selector(func_name, func_properties)
        
        # 3. 生成数据文件
        self.generate_function_data(func_properties)
        
        # 4. 更新GUI
        self.update_gui_function_list(func_name)
        
        print("测试函数添加完成!")
    
    def create_algorithm_variant(self, variant_name, variant_code):
        """创建算法变种"""
        variant_file = self.extensions_path / f"{variant_name}.py"
        
        with open(variant_file, 'w', encoding='utf-8') as f:
            f.write(variant_code)
        
        print(f"算法变种已创建: {variant_file}")
    
    def run_compatibility_test(self):
        """运行兼容性测试"""
        print("运行兼容性测试...")
        
        # 测试各个模块
        test_results = {
            'WFO.py': self.test_wfo_module(),
            'CEC2022.py': self.test_cec_module(),
            'WFO_exe.py': self.test_gui_module(),
            'Demo.py': self.test_demo_module()
        }
        
        # 输出测试结果
        for module, result in test_results.items():
            status = "通过" if result else "失败"
            print(f"{module}: {status}")
        
        return all(test_results.values())

# 使用示例
toolkit = ProjectExtensionToolkit()

# 备份项目
toolkit.backup_project()

# 添加新维度支持
toolkit.add_dimension_support([30, 50, 100])

# 添加新测试函数
new_func_properties = {
    'number': 13,
    'name': 'Styblinski-Tang',
    'dimensions': [2, 10, 20, 30, 50],
    'bounds': [-5, 5],
    'optimum': -39.16599 * 2  # 2维的理论最优值
}

toolkit.add_test_function('styblinski_tang', new_func_code, new_func_properties)

# 运行兼容性测试
if toolkit.run_compatibility_test():
    print("所有扩展成功集成!")
else:
    print("检测到兼容性问题，请检查错误日志。")
'''
    
    print("=== 项目扩展工具包 ===")
    print(toolkit_code)

if __name__ == "__main__":
    print("=== WFO项目扩展完整指南 ===\n")
    
    extender = ProjectExtender()
    
    # 1. 维度扩展
    print("1. 扩展维度支持:")
    extender.extend_dimension_support()
    
    print("\n" + "="*50 + "\n")
    
    # 2. 函数扩展
    print("2. 添加新测试函数:")
    extender.add_new_test_functions()
    
    print("\n" + "="*50 + "\n")
    
    # 3. 算法变种
    print("3. 创建算法变种:")
    extender.create_algorithm_variants()
    
    print("\n" + "="*50 + "\n")
    
    # 4. 性能优化
    print("4. 性能优化:")
    extender.performance_optimization()
    
    print("\n" + "="*50 + "\n")
    
    # 5. GUI增强
    print("5. GUI界面增强:")
    extender.gui_enhancement()
    
    print("\n" + "="*50 + "\n")
    
    # 6. 扩展工具包
    print("6. 扩展工具包:")
    create_extension_toolkit()
    
    print("\n=== 扩展指南完成 ===")
    print("按照以上指南，您可以系统地扩展WFO项目的功能。")
    print("建议在扩展前备份项目，并逐步进行测试。") 