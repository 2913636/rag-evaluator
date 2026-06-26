"""
RAG 质量评估器 — 核心引擎
=========================
LLM-as-Judge：用 LLM 给 RAG 回答打分的三维度评测。

维度：
  1. 忠实度 (Faithfulness) — 回答是否基于参考资料，有无编造？
  2. 相关性 (Relevance)    — 回答是否直接回应了用户问题？
  3. 完整性 (Completeness) — 回答是否覆盖了所有方面？

用法：
    from evaluator import RAGEvaluator
    eval = RAGEvaluator()
    result = eval.evaluate(question, answer, references)
    print(result["overall"])  # 综合评分 0-10
"""

import os
import json
import time
from typing import Optional

# 尝试导入 openai，不可用时给出友好提示
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


EVAL_PROMPT = """You are a RAG system quality evaluation expert. Evaluate the given answer based on three dimensions.

[Evaluation Dimensions]
1. Faithfulness (1-10): Is the answer based on the provided references? Any fabricated content?
2. Relevance (1-10): Does the answer directly address the user's question?
3. Completeness (1-10): Does the answer cover all aspects required by the question?

[Scoring Criteria]
- 8-10: Excellent, no obvious issues
- 5-7: Average, notable shortcomings
- 1-4: Poor, multiple errors or omissions

[Output Format]
Strictly output JSON only, no other text:
{
  "faithfulness": {"score": 0, "reason": "one sentence explanation"},
  "relevance": {"score": 0, "reason": "one sentence explanation"},
  "completeness": {"score": 0, "reason": "one sentence explanation"},
  "overall": 0,
  "improvements": ["suggestion 1", "suggestion 2"]
}"""


class RAGEvaluator:
    """
    RAG 质量评估器。

    支持两种模式：
      - live: 调用真实 LLM API（DeepSeek/GPT 等）
      - mock: 用规则打分（离线测试用）
    """

    def __init__(self, api_key: str = "", base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", mock: bool = False):
        """
        Args:
            api_key: LLM API Key（默认读 DEEPSEEK_KEY 环境变量）
            base_url: API 地址
            model: 模型名称
            mock: True=使用规则打分（不调 API），False=调用真实 LLM
        """
        self.mock = mock
        self.model = model

        if not mock:
            if OpenAI is None:
                raise ImportError("请安装 openai: pip install openai")
            key = api_key or os.getenv("DEEPSEEK_KEY", "")
            if not key:
                raise ValueError("请设置 DEEPSEEK_KEY 环境变量或传入 api_key 参数")
            self.client = OpenAI(api_key=key, base_url=base_url)
        else:
            self.client = None

        self._stats = {"total_evals": 0, "total_time_ms": 0, "total_cost_estimate": 0.0}

    # ── 核心评估 ────────────────────────

    def evaluate(self, question: str, answer: str,
                 references: list[str] = None) -> dict:
        """
        评估一次 RAG 回答。

        Args:
            question: 用户问题
            answer: RAG 系统给出的回答
            references: 检索到的参考文档（可选但强烈建议提供）

        Returns:
            {
                "faithfulness": {"score": int, "reason": str},
                "relevance": {"score": int, "reason": str},
                "completeness": {"score": int, "reason": str},
                "overall": int,
                "improvements": [str, ...],
                "cost_estimate": float,  # 预估 API 费用（美元）
                "duration_ms": int,
            }
        """
        t0 = time.perf_counter()

        if self.mock:
            result = self._mock_evaluate(question, answer, references)
        else:
            result = self._llm_evaluate(question, answer, references)

        elapsed = int((time.perf_counter() - t0) * 1000)
        result["duration_ms"] = elapsed

        self._stats["total_evals"] += 1
        self._stats["total_time_ms"] += elapsed

        return result

    def _llm_evaluate(self, question: str, answer: str,
                      references: list[str] = None) -> dict:
        """调用 LLM 进行评估"""
        # 构建参考资料文本
        ref_text = ""
        if references:
            ref_parts = []
            for i, ref in enumerate(references, 1):
                ref_parts.append(f"[Ref {i}] {ref}")
            ref_text = "\n".join(ref_parts)

        user_content = f"""User Question: {question}

{ref_text if ref_text else '(No references provided — evaluate based on general knowledge consistency)'}

[RAG System Answer]
{answer}

Please evaluate the above answer quality."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EVAL_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        # 估算费用（DeepSeek 约 $0.28/1M input tokens）
        usage = response.usage
        cost = (usage.prompt_tokens * 0.28 + usage.completion_tokens * 1.10) / 1_000_000
        result["cost_estimate"] = round(cost, 6)
        result["tokens_used"] = usage.total_tokens

        self._stats["total_cost_estimate"] += cost
        return result

    def _mock_evaluate(self, question: str, answer: str,
                       references: list[str] = None) -> dict:
        """
        规则打分（不调 API，用于快速测试）。

        简单的启发式规则：
          - 忠实度：检查回答中的数字/事实是否在参考文档中出现
          - 相关性：检查回答是否包含问题关键词
          - 完整性：检查回答长度和覆盖度
        """
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        ref_words = set()
        if references:
            for r in references:
                ref_words.update(r.lower().split())

        # 相关性：问题关键词在回答中的覆盖率
        q_covered = q_words & a_words
        relevance = min(10, max(1, int(len(q_covered) / max(len(q_words), 1) * 10)))

        # 忠实度：回答内容在参考文档中的比例
        if ref_words:
            a_in_ref = a_words & ref_words
            faithfulness = min(10, max(1, int(len(a_in_ref) / max(len(a_words), 1) * 12)))
        else:
            faithfulness = 5  # 无参考文档时默认为中

        # 完整性：回答长度
        completeness = min(10, max(1, len(answer) // 40))

        overall = round((faithfulness + relevance + completeness) / 3)

        return {
            "faithfulness": {
                "score": faithfulness,
                "reason": f"Mock: {len(a_words & ref_words) if ref_words else '?'} of {len(a_words)} answer words found in references"
            },
            "relevance": {
                "score": relevance,
                "reason": f"Mock: {len(q_covered)} of {len(q_words)} question keywords covered"
            },
            "completeness": {
                "score": completeness,
                "reason": f"Mock: answer length {len(answer)} chars"
            },
            "overall": overall,
            "improvements": ["(Mock mode — use real LLM for actionable suggestions)"],
            "cost_estimate": 0.0,
        }

    # ── 批量评估 ────────────────────────

    def evaluate_batch(self, test_cases: list[dict]) -> list[dict]:
        """
        批量评估多个测试用例。

        Args:
            test_cases: [{"question": str, "answer": str, "references": [str]}, ...]

        Returns:
            [result_dict, ...] 每个 result 包含原始用例信息 + 评分
        """
        results = []
        total = len(test_cases)
        for i, tc in enumerate(test_cases, 1):
            print(f"  [{i}/{total}] Evaluating: {tc['question'][:60]}...")
            result = self.evaluate(
                question=tc["question"],
                answer=tc["answer"],
                references=tc.get("references"),
            )
            result["_question"] = tc["question"]
            result["_answer"] = tc["answer"][:200]
            results.append(result)
        return results

    # ── 统计 ────────────────────────────

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def aggregate(self, results: list[dict]) -> dict:
        """
        聚合多轮评估结果。

        Returns:
            {
                "count": 总数,
                "avg_faithfulness": 均分,
                "avg_relevance": 均分,
                "avg_completeness": 均分,
                "avg_overall": 均分,
                "pass_rate": 综合 >= 7 的比例,
                "worst_cases": 最差的 3 个,
                "total_cost": 总费用,
                "total_time_ms": 总耗时,
            }
        """
        if not results:
            return {}

        n = len(results)
        f_scores = [r["faithfulness"]["score"] for r in results]
        r_scores = [r["relevance"]["score"] for r in results]
        c_scores = [r["completeness"]["score"] for r in results]
        o_scores = [r["overall"] for r in results]

        # 找出最差的 3 个
        ranked = sorted(results, key=lambda r: r["overall"])
        worst = []
        for r in ranked[:3]:
            worst.append({
                "question": r.get("_question", "?")[:80],
                "overall": r["overall"],
                "reason": r["faithfulness"]["reason"][:80],
            })

        return {
            "count": n,
            "avg_faithfulness": round(sum(f_scores) / n, 1),
            "avg_relevance": round(sum(r_scores) / n, 1),
            "avg_completeness": round(sum(c_scores) / n, 1),
            "avg_overall": round(sum(o_scores) / n, 1),
            "pass_rate": round(sum(1 for s in o_scores if s >= 7) / n * 100, 1),
            "worst_cases": worst,
            "total_cost": round(sum(r.get("cost_estimate", 0) for r in results), 4),
            "total_time_ms": sum(r.get("duration_ms", 0) for r in results),
        }
