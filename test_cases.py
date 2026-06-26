"""
测试用例生成与管理
==================
内置 12 条 RAG 评测用例，覆盖常见场景：
  - 事实检索、数字精确查询
  - 多跳推理、对比分析
  - 边界测试（无答案、矛盾信息）
  - 中英文混合

也可调用 LLM 自动生成更多用例。
"""

import json
from pathlib import Path

# ── 内置测试用例 ──────────────────────

BUILTIN_CASES = {
    "factual": [
        {
            "question": "特斯拉 Model 2 的售价和续航是多少？",
            "answer": "特斯拉 Model 2 售价 2.5 万美元，续航 500 公里，于 2026 年 Q1 开始交付。",
            "references": [
                "特斯拉 Model 2 售价 2.5 万美元，续航 500 公里。",
                "Model 2 是特斯拉最便宜的车型，2026 年 Q1 开始交付。",
            ],
        },
        {
            "question": "Python 的 GIL 是什么？",
            "answer": "GIL（全局解释器锁）是 CPython 中的一种机制，确保同一时刻只有一个线程执行 Python 字节码，以保护内存管理安全。",
            "references": [
                "GIL (Global Interpreter Lock) is a mutex that protects access to Python objects, "
                "preventing multiple threads from executing Python bytecode at once in CPython.",
            ],
        },
        {
            "question": "2026 年诺贝尔文学奖得主是谁？",
            "answer": "截至目前，2026 年诺贝尔文学奖尚未公布。",
            "references": [
                "2026 年诺贝尔奖各奖项将于 10 月陆续公布。",
            ],
        },
    ],
    "precision": [
        {
            "question": "比亚迪 2026 年 Q1 全球销量是多少？",
            "answer": "比亚迪 2026 年 Q1 全球销量突破 100 万辆，同比增长 59.8%。",
            "references": [
                "比亚迪 2026 年 Q1 全球销量突破 100 万辆，同比增长 59.8%，其中海外市场占比约 30%。",
            ],
        },
        {
            "question": "MCP 协议有哪三种传输方式？",
            "answer": "MCP 协议支持三种传输方式：stdio（标准输入输出）、SSE（Server-Sent Events）和 Streamable HTTP。",
            "references": [
                "MCP 协议定义三种传输方式：stdio、SSE（Server-Sent Events）、Streamable HTTP。",
                "stdio 适合本地工具，SSE 适合服务端推送，Streamable HTTP 适合无状态远程调用。",
            ],
        },
        {
            "question": "LangGraph 中 Checkpoint 的作用是什么？",
            "answer": "Checkpoint 在 LangGraph 中用于持久化状态图执行到每个节点的状态，支持暂停后恢复、人工审核中断、以及状态回滚。",
            "references": [
                "LangGraph Checkpoint 是状态图的状态快照机制，每经过一个节点自动存档，"
                "支持 interrupt_before 暂停、从断点恢复、以及状态回滚。",
            ],
        },
    ],
    "adversarial": [
        {
            "question": "苹果公司 2026 年推出了哪些新产品？",
            "answer": "苹果在 2026 年推出了 iPhone 18、Apple Watch Series 12 和 Apple Car 自动驾驶汽车。",
            "references": [
                "苹果公司 2026 年 6 月 WWDC 发布了 iPhone 18 和 Apple Watch Series 12。",
            ],
        },
        {
            "question": "Docker 和 Kubernetes 的区别是什么？",
            "answer": "Docker 是容器化技术，Kubernetes 是容器编排平台。Docker 负责打包，Kubernetes 负责管理。",
            "references": [
                "Docker 是容器运行时和镜像构建工具。",
                "Kubernetes 是容器编排平台，管理容器的部署、扩展和负载均衡。",
            ],
        },
    ],
    "edge_cases": [
        {
            "question": "如何给量子计算机安装 Windows？",
            "answer": "这个问题本身存在逻辑错误——量子计算机运行的是量子比特，不存在传统意义上的操作系统安装概念。当前量子计算机通过经典计算机进行控制和编程。",
            "references": [
                "量子计算机使用量子比特而非经典比特，目前无法运行传统操作系统。",
                "量子计算机通过经典计算机接口进行编程和控制。",
            ],
        },
        {
            "question": "请对比一下不存在的产品 A 和不存在的产品 B。",
            "answer": "产品 A 更好，因为它的性能更强。",
            "references": [
                "(无相关资料)",
            ],
        },
    ],
    "multi_hop": [
        {
            "question": "特斯拉 Model 2 的竞争对手是谁，他们的产品有什么优势？",
            "answer": "特斯拉 Model 2 的主要竞争对手是比亚迪海鸥。比亚迪海鸥 2026 年 Q1 全球销量突破 100 万辆，价格更低；Model 2 的优势在于品牌影响力和充电网络覆盖。",
            "references": [
                "特斯拉 Model 2 售价 2.5 万美元，续航 500 公里，定位入门级市场。",
                "比亚迪海鸥 2026 年 Q1 全球销量突破 100 万辆，起售价约 1.5 万美元。",
            ],
        },
        {
            "question": "RAG 系统中混合检索和纯向量检索各有什么优缺点？",
            "answer": "混合检索结合向量检索（语义相似）和 BM25（关键词匹配），通过 RRF 融合两种信号，对精确关键词查询召回率更高。纯向量检索对语义理解更好，但对数字、型号等精确匹配弱。",
            "references": [
                "向量检索擅长语义相似度匹配，但对专有名词、数字、日期等精确匹配效果差。",
                "BM25 基于词频和逆文档频率，对关键词精确匹配效果好。",
                "RRF（倒数排名融合）是一种无需调参的融合策略，对排名取倒数再求和排序。",
            ],
        },
    ],
}


def get_builtin_cases(categories: list[str] = None) -> list[dict]:
    """
    获取内置测试用例。

    Args:
        categories: 限定类别，None=全部。
                    可选：factual, precision, adversarial, edge_cases, multi_hop

    Returns:
        [{"question": ..., "answer": ..., "references": [...]}, ...]
    """
    if categories is None:
        categories = list(BUILTIN_CASES.keys())

    cases = []
    for cat in categories:
        if cat in BUILTIN_CASES:
            cases.extend(BUILTIN_CASES[cat])
    return cases


def get_expected_outcomes() -> dict:
    """
    获取每个用例的期望评测结果（用于验证评估器是否有效）。

    Returns:
        {question_prefix: {"expected_faithfulness": "high"|"low", "note": str}, ...}
    """
    return {
        "特斯拉 Model 2 的售价": {"expected_faithfulness": "high", "note": "回答与参考文献一致"},
        "苹果公司 2026 年推出": {"expected_faithfulness": "low", "note": "编造了 Apple Car"},
        "如何给量子计算机安装": {"expected_faithfulness": "high", "note": "正确识别问题逻辑错误"},
        "请对比一下不存在的产品": {"expected_faithfulness": "low", "note": "没有参考资料且回答敷衍"},
        "比亚迪 2026 年 Q1": {"expected_faithfulness": "high", "note": "数字精确匹配"},
        "RAG 系统中混合检索": {"expected_relevance": "high", "note": "完整回应了对比问题"},
    }


def save_cases(cases: list[dict], path: str = "test_cases.json"):
    """保存测试用例到 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


def load_cases(path: str = "test_cases.json") -> list[dict]:
    """从 JSON 文件加载测试用例"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_with_llm(evaluator, count: int = 10,
                       domain: str = "general") -> list[dict]:
    """
    用 LLM 自动生成更多测试用例。

    Args:
        evaluator: RAGEvaluator 实例（需要真实 LLM）
        count: 生成数量
        domain: 领域（general/tech/medical/legal）

    Returns:
        [{"question": ..., "answer": ..., "references": [...]}, ...]
    """
    if evaluator.mock:
        print("Mock 模式无法生成用例，请使用真实 LLM")
        return []

    prompt = f"""Generate {count} test cases for evaluating a RAG system in the {domain} domain.

Each test case should include:
1. A realistic user question
2. A RAG system answer (some good, some with hallucinations or errors)
3. Reference documents that contain the ground truth

Output strictly as JSON array:
[
  {{
    "question": "...",
    "answer": "...",
    "references": ["ref1", "ref2"]
  }}
]

Make the answers varied:
- Some accurate and well-grounded
- Some with subtle hallucinations (wrong numbers, dates)
- Some partially correct but incomplete
- Some completely off-topic"""

    response = evaluator.client.chat.completions.create(
        model=evaluator.model,
        messages=[
            {"role": "system", "content": "You are a test case generator. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    # 兼容不同 JSON 结构
    if isinstance(data, list):
        return data
    for key in data:
        if isinstance(data[key], list):
            return data[key]
    return []
