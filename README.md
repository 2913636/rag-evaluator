# RAG 质量自动评估器

> LLM-as-Judge：用 LLM 给 RAG 回答打分的三维度评测系统。

## 评估维度

| 维度 | 英文 | 评分标准 |
|------|------|------|
| 忠实度 | Faithfulness | 回答是否基于参考资料？有无编造？ |
| 相关性 | Relevance | 回答是否直接回应了用户问题？ |
| 完整性 | Completeness | 回答是否覆盖了所有方面？ |

每维度 1-10 分，综合评分 >= 7 为通过。

## 快速开始

```bash
# 1. 安装依赖
pip install openai

# 2. 设置 API Key
set DEEPSEEK_KEY=sk-your-key-here

# 3. 运行 Demo（内置 12 条测试用例）
py demo.py

# 4. 离线快速测试（规则打分，不调 API）
py demo.py --mock
```

## 项目结构

```
rag-evaluator/
├── evaluator.py     # 核心引擎：LLM-as-Judge + Mock 规则打分
├── test_cases.py    # 12 条内置用例 + 自动生成 + 导入导出
├── report.py        # 报告生成（控制台 + HTML）
├── demo.py          # 演示脚本
└── README.md
```

## 使用方式

```python
from evaluator import RAGEvaluator
from test_cases import get_builtin_cases
from report import print_console_report, generate_html_report

# 1. 初始化
evaluator = RAGEvaluator()

# 2. 加载用例
cases = get_builtin_cases(["factual", "precision"])

# 3. 批量评估
results = evaluator.evaluate_batch(cases)

# 4. 聚合统计
agg = evaluator.aggregate(results)
print(f"Avg Overall: {agg['avg_overall']}/10, Pass Rate: {agg['pass_rate']}%")

# 5. 生成报告
print_console_report(agg, results)
generate_html_report(agg, results)  # -> report.html
```

## 内置测试用例（12 条，5 类）

| 类别 | 数量 | 覆盖场景 |
|------|:--:|------|
| factual | 3 | 事实检索、概念解释、未公布信息 |
| precision | 3 | 数字精确、协议细节、机制原理 |
| adversarial | 2 | 编造检测、对比分析 |
| edge_cases | 2 | 逻辑错误问题、无参考资料 |
| multi_hop | 2 | 多跳推理、优缺点对比 |

