# WFO: 水流优化算法 (Water Flow Optimizer)

## 项目简介

WFO (Water Flow Optimizer) 是一个基于自然启发的进化算法，用于全局优化问题的求解。该算法模拟水流在自然环境中的层流和湍流行为，通过模拟水粒子的运动规律来寻找全局最优解。

## 算法原理

### 核心思想
水流优化算法基于水流的两种主要状态：
- **层流 (Laminar Flow)**: 水粒子有序流动，趋向于全局最优位置
- **湍流 (Turbulent Flow)**: 水粒子随机扰动，包含螺旋流和一般湍流

### 算法特点
- 全局搜索能力强
- 收敛速度快
- 参数设置简单
- 适用于多种优化问题

## 参考文献
Kaiping Luo. Water Flow Optimizer: a nature-inspired evolutionary algorithm for global optimization. IEEE Transactions on Cybernetics, 2021.

## 项目结构

```
WFO-Python/
├── 📁 核心模块
│   ├── WFO.py                      # 核心算法实现
│   ├── WFO_exe.py                  # GUI可执行界面 (主程序)
│   ├── Demo.py                     # 算法使用示例
│   ├── CEC2022.py                  # CEC2022基准测试函数集
│   ├── README.md                   # 项目说明文档
│   └── 项目结构说明.md              # 详细结构说明
├── 📁 docs/ (文档和指南)
│   ├── rosenbrock_详解.py          # Rosenbrock函数详细解析
│   ├── 自定义测试函数指南.py        # 创建自定义测试函数指南
│   ├── WFO算法详解.py             # WFO算法机制深度解析
│   ├── 项目扩展指南.py             # 项目功能扩展完整指南
│   └── 性能优化分析指南.py         # 性能优化和结果分析方法
├── 📁 examples/ (示例和扩展)
│   └── WFO_optimized_style.py      # WFO算法优化版本示例
├── 📁 build/ (构建工具)
│   └── pyinstaller.exe             # 打包配置文件
├── 📁 utils/ (实用工具)
│   ├── temp.txt                    # 临时结果文件
│   └── try.txt                     # 测试文件
├── 📁 icon/ (图标资源)
│   ├── image.png                   # 程序图标
│   └── Water flow.ico              # 窗口图标
└── 📁 input_data/ (基准测试数据)
    ├── M_*.txt                     # 旋转矩阵文件
    ├── shift_data_*.txt            # 位移向量文件
    ├── shuffle_data_*.txt          # 打乱数据文件
    └── Rand_Seeds.txt              # 随机种子文件
```

## 快速开始

### 🚀 基础使用
```bash
# 1. GUI界面 (推荐新手)
python WFO_exe.py

# 2. 命令行示例
python Demo.py

# 3. 学习算法原理
python docs/WFO算法详解.py
```

### 📚 学习路径
1. **入门**: 运行 `WFO_exe.py` 体验GUI界面
2. **理解**: 阅读 `docs/WFO算法详解.py` 了解算法原理
3. **实践**: 参考 `Demo.py` 编写自己的优化代码
4. **进阶**: 学习 `docs/` 目录下的各种指南
5. **扩展**: 使用 `docs/项目扩展指南.py` 添加新功能

## 功能特性

### 1. 核心算法模块 (WFO.py)
```python
def WFO(alg, prob):
    """
    水流优化算法主函数
    
    参数:
    alg: 算法参数类
        - NP: 水粒子数量
        - max_nfe: 最大函数评估次数
        - pl: 层流概率
        - pe: 涡流概率
    
    prob: 问题参数类
        - lb: 下界向量
        - ub: 上界向量
        - fobj: 目标函数
        - dim: 问题维度
    
    返回:
    fb: 最优函数值
    xb: 最优解
    con: 收敛曲线
    """
```

### 2. 图形用户界面 (WFO_exe.py)
- **欢迎界面**: 项目介绍和开始按钮
- **主界面**: 包含以下功能模块
  - 目标函数选择 (CEC2022基准函数或自定义函数)
  - 优化准则设置 (最小化/最大化)
  - 算法参数配置
  - 问题参数设置
  - 实时结果显示
  - 收敛曲线绘制

### 3. 基准测试模块 (CEC2022.py)
提供CEC2022竞赛的12个基准测试函数：
- F1: Zakharov函数
- F2: Rosenbrock函数
- F3: Schaffer's F7函数
- F4: Non-Continuous Rastrigin函数
- F5: Levy函数
- F6: Bent Cigar函数
- F7: Hybrid函数1 (N=3)
- F8: Hybrid函数2 (N=6) 
- F9: Hybrid函数3 (N=5)
- F10: Composition函数1 (N=5)
- F11: Composition函数2 (N=4)
- F12: Composition函数3 (N=6)

### 4. 演示程序 (Demo.py)
```python
# 算法参数设置
class alg:
    NP = 20          # 种群大小
    max_nfe = 10000  # 最大评估次数
    pl = 0.7         # 层流概率
    pe = 0.3         # 涡流概率

# 问题参数设置
class prob:
    dim = 10                    # 维度
    lb = [-100] * dim          # 下界
    ub = [100] * dim           # 上界
    def fobj(x):               # 目标函数
        return cec22_test_func(x, dim, 1, 2)

# 运行算法
fb, xb, con = WFO(alg, prob)
```

## 安装和使用

### 环境要求
- Python 3.6+
- NumPy
- Matplotlib
- Tkinter (通常随Python安装)

### 安装依赖
```bash
pip install numpy matplotlib
```

### 使用方法

#### 1. GUI界面使用
直接运行主程序：
```bash
python WFO_exe.py
```

操作步骤：
1. 点击"Start"进入主界面
2. 选择目标函数类型 (CEC2022或自定义)
3. 设置问题参数 (维度、上下界)
4. 配置算法参数 (种群大小、最大评估次数等)
5. 点击"Solve"开始优化
6. 查看结果和收敛曲线

#### 2. 编程调用
```python
from WFO import WFO
from CEC2022 import cec22_test_func

# 设置算法参数
class alg:
    NP = 30
    max_nfe = 15000
    pl = 0.7
    pe = 0.3

# 设置问题参数  
class prob:
    dim = 20
    lb = [-100] * dim
    ub = [100] * dim
    def fobj(x):
        # 使用CEC2022函数1
        return cec22_test_func(x, dim, 1, 1)

# 运行优化
best_value, best_solution, convergence = WFO(alg, prob)

print(f"最优值: {best_value}")
print(f"最优解: {best_solution}")
```

#### 3. 自定义目标函数
```python
# 示例1: 球函数
def sphere_function(x):
    return sum(xi**2 for xi in x)

# 示例2: Rastrigin函数
import math
def rastrigin_function(x):
    n = len(x)
    return sum(xi**2 - 10*math.cos(2*math.pi*xi) for xi in x) + 10*n

# 使用自定义函数
class prob:
    dim = 10
    lb = [-5.12] * dim
    ub = [5.12] * dim
    fobj = rastrigin_function
```

## 算法参数说明

### 关键参数
- **NP (种群大小)**: 控制搜索的并行度，通常设置为20-50
- **max_nfe (最大评估次数)**: 终止条件，根据问题复杂度设置
- **pl (层流概率)**: 控制全局搜索强度，推荐0.6-0.8
- **pe (涡流概率)**: 控制局部搜索强度，推荐0.2-0.4

### 参数调优建议
- 对于单峰函数：增大pl值，减小pe值
- 对于多峰函数：适当减小pl值，增大pe值
- 高维问题：增加NP和max_nfe
- 低维问题：可适当减小NP

## 实验结果示例

### CEC2022基准测试结果
在标准测试条件下 (D=10, NP=20, max_nfe=10000)：

| 函数 | 最优值 | 平均值 | 标准差 |
|------|--------|--------|--------|
| F1   | 1.2e-15| 2.3e-12| 5.1e-12|
| F2   | 8.9e+00| 2.1e+01| 1.5e+01|
| F3   | 2.3e-01| 3.4e-01| 1.2e-01|
| ...  | ...    | ...    | ...    |

### 收敛性能分析
- 收敛速度：前期快速收敛，后期精细搜索
- 稳定性：多次运行结果稳定
- 可扩展性：支持2-100维问题优化

## 常见问题解答

### Q1: 算法不收敛怎么办？
- 增加max_nfe值
- 调整pl和pe参数
- 检查目标函数定义是否正确

### Q2: 如何处理约束优化问题？
- 使用罚函数法
- 在目标函数中添加约束违反惩罚项

### Q3: 如何提高算法性能？
- 根据问题特性调整参数
- 增加种群大小NP
- 使用问题相关的初始化策略

## 开发团队

**作者**: 罗开平 (Kaiping Luo)  
**单位**: 北京航空航天大学  
**邮箱**: kaipingluo@buaa.edu.cn  

## 许可证

本项目遵循学术研究使用协议，如需商业使用请联系作者。

## 更新日志

### v1.0 (2022)
- 实现WFO核心算法
- 添加CEC2022基准测试函数
- 开发GUI界面
- 提供使用示例

## 致谢

感谢北京航空航天大学对本项目的支持，以及IEEE Transactions on Cybernetics期刊对算法论文的发表。

---

如有任何问题或建议，欢迎通过邮箱联系作者或在项目仓库中提交Issue。 