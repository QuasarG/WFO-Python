import numpy as np

class WaterFlowOptimizer:
    """
    水流优化器 (WFO) 的主类。
    
    该类封装了WFO算法的全部逻辑，包括初始化、层流与湍流模拟，
    以及迭代寻优的全过程。
    """

    def __init__(self, obj_fun, dim, bounds, pop_size=30, max_iter=500, pl=0.2, pe=0.8):
        """
        算法的构造函数，用于初始化所有参数和种群。
        """
        # 存储算法配置
        self.obj_fun = obj_fun
        self.dim = dim
        self.lower_bound, self.upper_bound = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.laminar_prob = pl
        self.eddying_prob = pe

        # 初始化种群（水分子群）
        # 在多维搜索空间中，随机生成所有水分子（解）的初始位置
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        
        # 计算初始适应度（势能）
        # 对每个水分子，调用目标函数计算其初始的函数值
        self.fitness = np.array([self.obj_fun(ind) for ind in self.population])

        # 记录并存储初始的最优解
        best_idx = np.argmin(self.fitness)
        self.gbest_fitness = self.fitness[best_idx]
        self.gbest_position = self.population[best_idx].copy()
        
        # 用于记录历史最优值的列表
        self.fitness_history = []

    def run(self):
        """
        启动WFO算法的主执行循环。
        返回一个包含(全局最优解, 全局最优适应度)的元组。
        """
        # 主迭代循环，模拟水流的演进过程
        for t in range(self.max_iter):
            # 创建一个当前种群的副本，用于存放本轮产生的新位置
            new_population = self.population.copy()

            # 核心决策：根据层流概率决定水流形态
            if np.random.rand() < self.laminar_prob:
                self._laminar_flow(new_population)
            else:
                self._turbulent_flow(new_population)

            # 边界处理
            # 将所有超出边界的变量值拉回到边界上
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)

            # 评估与更新
            self._evaluate_and_update(new_population)

            # 记录当前迭代的全局最优适应度
            self.fitness_history.append(self.gbest_fitness)
            
            # 打印迭代进度
            if (t + 1) % 100 == 0:
                print(f"迭代 {t + 1}/{self.max_iter}, 当前最优值: {self.gbest_fitness:.6f}")

        print("WFO算法运行结束。")
        return self.gbest_position, self.gbest_fitness

    def _laminar_flow(self, new_pop):
        """模拟层流：所有水分子向着一个有希望的方向汇集（开采）。"""
        # --- 确定有希望的方向向量 d ---
        # 从非最优解中随机选择一个体 k
        non_gbest_indices = np.delete(np.arange(self.pop_size), np.argmin(self.fitness))
        k_idx = np.random.choice(non_gbest_indices, 1)[0]
        
        # 方向向量 d 指向全局最优解
        direction_vec = self.gbest_position - self.population[k_idx]

        # --- 所有水分子沿该方向移动 ---
        for i in range(self.pop_size):
            step_size = np.random.rand()
            new_pop[i] = self.population[i] + step_size * direction_vec

    def _turbulent_flow(self, new_pop):
        """模拟湍流：通过局部扰动增加种群多样性（探索）。"""
        for i in range(self.pop_size):
            # 随机选择一个“冲撞”伙伴 k
            k_idx = np.random.choice(np.delete(np.arange(self.pop_size), i), 1)[0]
            # 随机选择一个维度 j1 进行扰动
            j1 = np.random.randint(0, self.dim)
            
            # 根据漩涡概率决定扰动方式
            if np.random.rand() < self.eddying_prob:
                # 执行“漩涡”操作 (Eddying)
                rho = np.abs(self.population[i, j1] - self.population[k_idx, j1])
                theta = np.random.uniform(-np.pi, np.pi)
                new_pop[i, j1] = self.population[i, j1] + rho * theta * np.cos(theta)
            else:
                # 执行“跨层运动”操作 (Over-layer moving)
                j2 = np.random.choice(np.delete(np.arange(self.dim), j1), 1)[0]
                scale_factor = (self.population[k_idx, j2] - self.lower_bound) / (self.upper_bound - self.lower_bound)
                new_pop[i, j1] = self.lower_bound + (self.upper_bound - self.lower_bound) * scale_factor

    def _evaluate_and_update(self, new_pop):
        """评估新种群，并根据优劣决定是否更新。"""
        # 计算所有新位置的适应度
        new_fitness = np.array([self.obj_fun(ind) for ind in new_pop])
        
        # 贪心选择
        # 使用布尔索引，找到新解优于旧解的位置
        improvement_mask = new_fitness < self.fitness
        
        # 仅用新解替换掉那些变得更好的旧解
        self.fitness[improvement_mask] = new_fitness[improvement_mask]
        self.population[improvement_mask] = new_pop[improvement_mask]

        # 更新全局最优解
        current_best_fitness = np.min(self.fitness)
        if current_best_fitness < self.gbest_fitness:
            self.gbest_fitness = current_best_fitness
            best_idx = np.argmin(self.fitness)
            self.gbest_position = self.population[best_idx].copy()

if __name__ == '__main__':

    # 1. 定义一个需要优化的目标函数 (以Sphere函数为例)
    #    目标：找到一组(x1, x2, ..., xn)，使得 x1^2 + x2^2 + ... 的值最小
    def sphere_function(x):
        return np.sum(x**2)

    # 2. 设置问题的参数
    problem_dim = 10         # 问题的变量个数
    search_bounds = (-100, 100) # 变量的取值范围

    # 3. 创建WFO算法实例
    wfo_solver = WaterFlowOptimizer(
        obj_fun=sphere_function,
        dim=problem_dim,
        bounds=search_bounds,
        pop_size=50,
        max_iter=1000
    )

    # 4. 运行算法并获取结果
    best_solution, best_value = wfo_solver.run()

    # 5. 打印最终结果
    print("\n----------------- 最终结果 -----------------")
    print(f"找到的最优解 (位置): {best_solution}")
    print(f"对应的最优值 (函数最小值): {best_value}")

    # 可选：绘制适应度进化曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(wfo_solver.fitness_history)
        plt.title("WFO 算法收敛曲线")
        plt.xlabel("迭代次数")
        plt.ylabel("最优适应度值")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\n提示: 未安装matplotlib库，无法绘制收敛曲线。可使用 'pip install matplotlib' 安装。")