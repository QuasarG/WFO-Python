"""
WFO性能优化和测试结果分析指南
===============================

本指南提供系统化的性能优化策略和结果分析方法。
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import stats

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.results = []
        self.benchmarks = {}
    
    def analyze_convergence(self, convergence_curve):
        """分析收敛特性"""
        
        analysis = {
            'final_value': convergence_curve[-1],
            'convergence_speed': self.calculate_convergence_speed(convergence_curve),
            'stability': self.calculate_stability(convergence_curve),
            'early_stop_point': self.find_early_stop_point(convergence_curve)
        }
        
        print(f"=== 收敛性分析 ===")
        print(f"最终值: {analysis['final_value']:.6e}")
        print(f"收敛速度: {analysis['convergence_speed']:.3f}")
        print(f"稳定性: {analysis['stability']:.3f}")
        print(f"建议提前停止点: {analysis['early_stop_point']}")
        
        return analysis
    
    def calculate_convergence_speed(self, curve):
        """计算收敛速度"""
        if len(curve) < 10:
            return 0.0
        
        # 计算前10%到后10%的改进率
        early_avg = np.mean(curve[:len(curve)//10])
        late_avg = np.mean(curve[-len(curve)//10:])
        
        improvement_rate = (early_avg - late_avg) / early_avg if early_avg != 0 else 0
        return improvement_rate
    
    def calculate_stability(self, curve):
        """计算收敛稳定性"""
        if len(curve) < 20:
            return 1.0
        
        # 分析后半段的变异系数
        latter_half = curve[len(curve)//2:]
        cv = np.std(latter_half) / np.mean(latter_half) if np.mean(latter_half) != 0 else 0
        stability = 1.0 / (1.0 + cv)  # 稳定性越高，变异系数越小
        
        return stability
    
    def benchmark_algorithms(self, algorithms, test_functions, runs=30):
        """算法基准测试"""
        
        results = {}
        
        for alg_name, algorithm in algorithms.items():
            results[alg_name] = {}
            
            for func_name, test_func in test_functions.items():
                print(f"测试 {alg_name} on {func_name}...")
                
                run_results = []
                for run in range(runs):
                    start_time = time.time()
                    best_value, best_solution, convergence = algorithm(test_func)
                    run_time = time.time() - start_time
                    
                    run_results.append({
                        'best_value': best_value,
                        'run_time': run_time,
                        'convergence': convergence
                    })
                
                # 统计分析
                best_values = [r['best_value'] for r in run_results]
                run_times = [r['run_time'] for r in run_results]
                
                results[alg_name][func_name] = {
                    'mean': np.mean(best_values),
                    'std': np.std(best_values),
                    'best': np.min(best_values),
                    'worst': np.max(best_values),
                    'median': np.median(best_values),
                    'mean_time': np.mean(run_times),
                    'success_rate': self.calculate_success_rate(best_values, test_func.optimum)
                }
        
        return results
    
    def statistical_test(self, results1, results2, alpha=0.05):
        """统计显著性检验"""
        
        # Wilcoxon符号秩检验
        statistic, p_value = stats.wilcoxon(results1, results2)
        
        print(f"=== 统计检验结果 ===")
        print(f"Wilcoxon统计量: {statistic}")
        print(f"p值: {p_value:.6f}")
        print(f"显著性水平: {alpha}")
        
        if p_value < alpha:
            print("结果具有统计显著性差异")
        else:
            print("结果无显著性差异")
        
        return p_value < alpha

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.optimization_strategies = {
            'memory': self.optimize_memory,
            'computation': self.optimize_computation,
            'algorithm': self.optimize_algorithm,
            'parallel': self.optimize_parallel
        }
    
    def optimize_memory(self, wfo_instance):
        """内存优化策略"""
        print("=== 内存优化 ===")
        print("1. 使用float32替代float64")
        print("2. 原地操作减少内存分配")
        print("3. 删除不必要的中间变量")
        print("4. 使用生成器处理大数据集")
        
        optimized_code = '''
def memory_optimized_wfo(alg, prob):
    # 使用float32
    X = np.zeros((alg.NP, prob.dim), dtype=np.float32)
    F = np.zeros(alg.NP, dtype=np.float32)
    
    # 原地初始化
    for i in range(alg.NP):
        X[i] = np.random.uniform(prob.lb, prob.ub, prob.dim).astype(np.float32)
        F[i] = prob.fobj(X[i])
    
    # 原地更新，避免创建新数组
    Y = np.empty_like(X)
    
    return X, F
'''
        return optimized_code
    
    def optimize_computation(self, wfo_instance):
        """计算优化策略"""
        print("=== 计算优化 ===")
        print("1. 矢量化操作")
        print("2. 避免重复计算")
        print("3. 使用NumPy内置函数")
        print("4. JIT编译优化")
        
        optimized_code = '''
from numba import jit

@jit(nopython=True)
def fast_laminar_flow(X, xb, pl):
    NP, dim = X.shape
    Y = np.empty_like(X)
    
    if np.random.random() < pl:
        k = np.random.randint(0, NP)
        for i in range(NP):
            r = np.random.random()
            for j in range(dim):
                Y[i, j] = X[i, j] + r * (xb[j] - X[k, j])
    
    return Y
'''
        return optimized_code
    
    def parameter_tuning_guide(self):
        """参数调优指南"""
        
        guide = """
=== 参数调优指南 ===

1. 基础参数设置：
   - NP (种群大小): 20-100，推荐 30-50
   - max_nfe: 根据问题复杂度，通常 dim × 1000-10000
   - pl (层流概率): 0.6-0.8，高维问题可适当降低
   - pe (涡流概率): 0.2-0.4，多峰问题可适当提高

2. 自适应策略：
   
def adaptive_parameters(iteration, max_iter, current_best):
    progress = iteration / max_iter
    
    # 动态调整层流概率
    pl = 0.8 - 0.3 * progress  # 从0.8线性降到0.5
    
    # 动态调整涡流概率
    pe = 0.2 + 0.2 * progress  # 从0.2线性升到0.4
    
    return pl, pe

3. 问题特定调优：
   - 单峰问题: 高pl (0.8), 低pe (0.2)
   - 多峰问题: 中pl (0.6), 高pe (0.4)
   - 高维问题: 增大NP, 降低pl
   - 时间敏感: 减小max_nfe, 使用early stopping

4. 网格搜索示例：
   
def grid_search_parameters(test_function, param_ranges):
    best_params = None
    best_performance = float('inf')
    
    for pl in param_ranges['pl']:
        for pe in param_ranges['pe']:
            for NP in param_ranges['NP']:
                # 测试当前参数组合
                performance = test_with_params(test_function, pl, pe, NP)
                
                if performance < best_performance:
                    best_performance = performance
                    best_params = {'pl': pl, 'pe': pe, 'NP': NP}
    
    return best_params, best_performance
"""
        print(guide)

def create_analysis_toolkit():
    """创建分析工具包"""
    
    toolkit_code = '''
class ComprehensiveAnalyzer:
    """综合分析工具"""
    
    def __init__(self):
        self.results_db = []
    
    def run_comprehensive_test(self, algorithm, test_suite, config):
        """运行综合测试"""
        
        results = {
            'convergence_analysis': {},
            'parameter_sensitivity': {},
            'scalability_analysis': {},
            'robustness_test': {}
        }
        
        # 1. 收敛性分析
        for func_name, test_func in test_suite.items():
            conv_results = []
            for run in range(config['runs']):
                _, _, convergence = algorithm(test_func)
                conv_results.append(convergence)
            
            results['convergence_analysis'][func_name] = {
                'mean_convergence': np.mean(conv_results, axis=0),
                'std_convergence': np.std(conv_results, axis=0),
                'final_values': [c[-1] for c in conv_results]
            }
        
        # 2. 参数敏感性分析
        param_ranges = {
            'pl': [0.5, 0.6, 0.7, 0.8, 0.9],
            'pe': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        results['parameter_sensitivity'] = self.parameter_sensitivity_analysis(
            algorithm, test_suite, param_ranges
        )
        
        # 3. 可扩展性分析
        dimensions = [10, 20, 30, 50, 100]
        results['scalability_analysis'] = self.scalability_analysis(
            algorithm, dimensions
        )
        
        # 4. 鲁棒性测试
        noise_levels = [0.0, 0.01, 0.05, 0.1]
        results['robustness_test'] = self.robustness_analysis(
            algorithm, test_suite, noise_levels
        )
        
        return results
    
    def generate_report(self, results):
        """生成分析报告"""
        
        report = f"""
=== WFO算法性能分析报告 ===

1. 收敛性能：
   - 平均收敛代数: {self.calculate_avg_convergence_gen(results)}
   - 收敛稳定性: {self.calculate_convergence_stability(results)}
   - 最终精度: {self.calculate_final_accuracy(results)}

2. 参数敏感性：
   - 最敏感参数: {self.find_most_sensitive_param(results)}
   - 推荐参数设置: {self.recommend_parameters(results)}

3. 可扩展性：
   - 维度扩展性能: {self.evaluate_scalability(results)}
   - 内存使用效率: {self.evaluate_memory_efficiency(results)}

4. 鲁棒性：
   - 噪声抗性: {self.evaluate_noise_resistance(results)}
   - 稳定性评分: {self.calculate_robustness_score(results)}

5. 改进建议：
{self.generate_improvement_suggestions(results)}
"""
        
        return report
'''
    
    print("=== 综合分析工具包 ===")
    print(toolkit_code)

if __name__ == "__main__":
    print("=== WFO性能优化和分析指南 ===\n")
    
    # 1. 性能分析
    analyzer = PerformanceAnalyzer()
    
    # 示例收敛曲线分析
    sample_curve = np.logspace(2, 0, 1000)  # 模拟收敛曲线
    analyzer.analyze_convergence(sample_curve)
    
    print("\n" + "="*40 + "\n")
    
    # 2. 性能优化
    optimizer = PerformanceOptimizer()
    optimizer.optimize_memory(None)
    optimizer.optimize_computation(None)
    
    print("\n" + "="*40 + "\n")
    
    # 3. 参数调优
    optimizer.parameter_tuning_guide()
    
    print("\n" + "="*40 + "\n")
    
    # 4. 分析工具包
    create_analysis_toolkit()
    
    print("\n=== 指南完成 ===")
    print("使用这些方法可以全面优化和分析WFO算法性能。") 