"""
RAG 质量自动评估器
===================
用 LLM-as-Judge 评估 RAG 回答的三个维度：忠实度、相关性、完整性。
输出 0-10 评分 + 改进建议。

运行：py evaluator.py
"""

import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_KEY", ""),
    base_url="https://api.deepseek.com"
)

# ═══ 评估 Prompt ═══

EVAL_PROMPT = """你是 RAG 系统质量评审专家。根据以下三个维度，评估给定回答的质量。

【评估维度】
1. 忠实度（1-10）：回答是否基于参考资料，有没有编造内容？
2. 相关性（1-10）：回答是否直接回应了用户问题？
3. 完整性（1-10）：回答是否覆盖了问题要求的所有方面？

【评分标准】
- 8-10：优秀，无明显问题
- 5-7：一般，有明显不足
- 1-4：很差，多处错误或遗漏

【输出格式】
严格输出 JSON，不要其他内容：
{
  "faithfulness": {"score": 0, "reason": "一句话说明"},
  "relevance": {"score": 0, "reason": "一句话说明"},
  "completeness": {"score": 0, "reason": "一句话说明"},
  "overall": 0,
  "improvements": ["改进建议1", "改进建议2"]
}"""


def evaluate(question: str, answer: str, references: list[str] = None) -> dict:
    """
    评估 RAG 回答质量。

    参数:
        question: 用户问题
        answer: RAG 系统给出的回答
        references: 检索到的参考文档列表

    返回:
        dict: 包含 faithfulness/relevance/completeness 评分和理由
    """
    ref_text = ""
    if references:
        ref_text = "【参考资料】\n" + "\n---\n".join(references)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "system",
            "content": EVAL_PROMPT
        }, {
            "role": "user",
            "content": f"""用户问题：{question}

{ref_text}

【RAG 系统回答】
{answer}

请评估以上回答的质量。"""
        }],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def print_report(result: dict):
    """格式化打印评估报告"""
    def bar(score):
        return "█" * score + "░" * (10 - score)

    print("\n" + "=" * 55)
    print("📊 RAG 质量评估报告")
    print("=" * 55)

    for key, label in [("faithfulness", "忠实度"), ("relevance", "相关性"), ("completeness", "完整性")]:
        s = result[key]["score"]
        print(f"\n{label}: {s}/10 {bar(s)}")
        print(f"  {result[key]['reason']}")

    overall = result.get("overall", 0)
    print(f"\n{'─' * 55}")
    print(f"📈 综合评分: {overall}/10 {'⭐' * min(5, overall // 2)}")

    improvements = result.get("improvements", [])
    if improvements:
        print(f"\n💡 改进建议:")
        for i, imp in enumerate(improvements, 1):
            print(f"  {i}. {imp}")
    print("=" * 55)


if __name__ == "__main__":
    print("=" * 55)
    print("🔍 RAG 质量自动评估器")
    print("=" * 55)
    print("用 LLM-as-Judge 评估回答质量（忠实度/相关性/完整性）")
    print()

    # Demo：模拟一个 RAG 系统的回答
    question = "特斯拉 Model 2 的售价和续航是多少？"

    references = [
        "特斯拉 Model 2 售价 2.5 万美元，续航 500 公里。",
        "Model 2 是特斯拉最便宜的车型，2026 年 Q1 开始交付。"
    ]

    # 场景1：好回答
    good_answer = "特斯拉 Model 2 售价 2.5 万美元，续航 500 公里，于 2026 年 Q1 开始交付。"

    # 场景2：差回答（编造+遗漏）
    bad_answer = "特斯拉 Model 2 售价 2 万美元，采用固态电池，续航 800 公里。"

    print("── 场景1：正确回答 ──")
    print(f"问题: {question}")
    print(f"回答: {good_answer}")
    result = evaluate(question, good_answer, references)
    print_report(result)

    print("\n\n── 场景2：编造+遗漏回答 ──")
    print(f"问题: {question}")
    print(f"回答: {bad_answer}")
    result = evaluate(question, bad_answer, references)
    print_report(result)
