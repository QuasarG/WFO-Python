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
├── 📁 examples/ (示例和扩展)
│   └── WFO_optimized_style.py      # 算法优化版本示例
├── 📁 build/ (构建工具)
│   └── pyinstaller.exe             # 打包工具
├── 📁 utils/ (实用工具)
├── 📁 icon/ (图标资源)
│   └── WFO.ico                     # 程序图标
└── 📁 input_data/ (测试数据)
    ├── shift_data_*.txt            # 函数位移数据
    ├── M_*_D*.txt                  # 旋转矩阵数据
    └── shuffle_data_*.txt          # 维度重排数据
```

## 快速开始

### 方法1: GUI界面 (推荐)
```bash
python WFO_exe.py
```
- 友好的图形界面
- 参数可视化设置
- 实时结果显示
- 收敛曲线绘制

### 方法2: 代码调用
```python
from WFO import WFO
from CEC2022 import cec22_test_func

# 算法参数设置
class alg:
    NP = 30          # 种群大小
    max_nfe = 15000  # 最大函数评估次数
    pl = 0.7         # 层流概率
    pe = 0.3         # 涡流概率

# 问题定义
n = 10  # 维度
class prob:
    dim = n
    lb = [-100] * n   # 下界
    ub = [100] * n    # 上界
    
    def fobj(x):      # 目标函数
        return cec22_test_func(x=x, nx=n, mx=1, func_num=1)[0]

# 运行优化
best_fitness, best_solution, convergence = WFO(alg, prob)

print(f'最优值: {best_fitness}')
print(f'最优解: {best_solution}')
```

## 核心功能

### 🧠 WFO算法 (`WFO.py`)
- 水流优化算法核心实现
- 层流和湍流模式自动切换
- 支持连续变量优化问题

### 🎯 测试函数集 (`CEC2022.py`)
- 12个CEC2022标准测试函数
- 支持2维、10维、20维测试
- 包含单峰、多峰、混合、组合等多种类型

### 🖥️ 图形界面 (`WFO_exe.py`)
- 直观的参数设置界面
- 函数选择和维度配置
- 实时优化过程显示
- 结果分析和可视化

### 📊 使用示例 (`Demo.py`)
- 基本使用方法演示
- 参数设置指导
- 结果处理示例

## 算法参数说明

| 参数 | 含义 | 建议范围 | 默认值 |
|------|------|----------|--------|
| NP | 种群大小 | 20-100 | 30 |
| max_nfe | 最大函数评估次数 | 5000-50000 | 15000 |
| pl | 层流概率 | 0.5-0.9 | 0.7 |
| pe | 涡流概率 | 0.1-0.5 | 0.3 |

## CEC2022测试函数

| 函数编号 | 函数名称 | 类型 | 特点 |
|----------|----------|------|------|
| F1 | Shifted Zakharov | 单峰 | 简单二次函数 |
| F2 | Shifted Rosenbrock | 多峰 | 经典测试函数 |
| F3 | Shifted Expanded Schaffer's F6 | 多峰 | 高频振荡 |
| F4 | Shifted Non-Continuous Rastrigin | 多峰 | 非连续性 |
| F5 | Shifted Levy | 多峰 | 复杂地形 |
| F6-F8 | 混合函数 | 混合 | 多函数组合 |
| F9-F12 | 组合函数 | 组合 | 加权多函数 |

## 环境要求

### 必需依赖
- Python 3.6+
- NumPy
- Matplotlib

### 安装命令
```bash
pip install numpy matplotlib
```

### 可选依赖
- Tkinter (GUI支持，通常随Python安装)
- Pillow (图像处理)

## 使用技巧

### 参数调优建议
1. **维度较低 (≤10)**：使用较小种群 (NP=20-30)
2. **维度较高 (>10)**：增加种群大小 (NP=50-100)
3. **收敛缓慢**：增加max_nfe或调整pl/pe比例
4. **过早收敛**：降低pl，增加pe

### 性能优化
1. **函数评估**：优化目标函数计算效率
2. **维度选择**：根据问题特点选择合适维度
3. **多次运行**：进行多次独立运行取平均结果

## 扩展开发

### 添加新测试函数
在`CEC2022.py`中按照现有格式添加新函数：
```python
def my_function(x, nx, Os, Mr, s_flag, r_flag):
    # 实现新的测试函数
    pass
```

### 算法改进
在`WFO.py`中修改核心逻辑：
- 层流/湍流模式改进
- 新的参数自适应策略
- 混合其他优化算法

### GUI增强
在`WFO_exe.py`中添加新功能：
- 高级参数设置
- 结果分析工具
- 批量测试功能

## 许可证
本项目基于开源协议，欢迎学术研究和教学使用。

## 引用方式
如果在研究中使用了本项目，请引用：
```
Kaiping Luo. Water Flow Optimizer: a nature-inspired evolutionary algorithm 
for global optimization. IEEE Transactions on Cybernetics, 2021.
```

## 联系方式
如有问题或建议，欢迎通过GitHub Issues反馈。 