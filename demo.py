"""
RAG 质量评估器 — Demo
======================
用内置 12 条测试用例，演示完整的 RAG 评测流程：
  1. 加载测试用例
  2. 逐条 LLM-as-Judge 评估
  3. 聚合统计
  4. 生成控制台报告 + HTML 报告

运行：
    py demo.py                    # 使用真实 LLM（需设置 DEEPSEEK_KEY）
    py demo.py --mock             # 使用规则打分（不调 API，快速测试）
"""

import sys
from evaluator import RAGEvaluator
from test_cases import get_builtin_cases
from report import print_console_report, generate_html_report


def main():
    mock_mode = "--mock" in sys.argv

    print("=" * 58)
    print("  RAG Quality Evaluator")
    print("  LLM-as-Judge | Faithfulness + Relevance + Completeness")
    print("=" * 58)

    # ── 1. 初始化 ──
    if mock_mode:
        print("\n  Mode: Mock (rule-based scoring, no API call)")
        evaluator = RAGEvaluator(mock=True)
    else:
        print("\n  Mode: Live (DeepSeek LLM-as-Judge)")
        try:
            evaluator = RAGEvaluator()
        except ValueError as e:
            print(f"\n  [!] {e}")
            print("  Tip: set DEEPSEEK_KEY env var, or run with --mock")
            sys.exit(1)

    # ── 2. 加载测试用例 ──L
    all_cases = get_builtin_cases()
    print(f"  Loaded {len(all_cases)} test cases (5 categories)")

    # 分类统计
    from test_cases import BUILTIN_CASES
    for cat, cases in BUILTIN_CASES.items():
        print(f"    - {cat}: {len(cases)} cases")

    # ── 3. 批量评估 ──
    print(f"\n  Running evaluation...\n")
    results = evaluator.evaluate_batch(all_cases)

    # ── 4. 聚合统计 ──
    aggregate = evaluator.aggregate(results)

    # ── 5. 报告 ──
    print_console_report(aggregate, results, "RAG Quality Evaluation Report")

    html_path = generate_html_report(aggregate, results)
    print(f"\n  HTML report saved to: {html_path}")

    # ── 6. 评测器本身验证 ──
    if not mock_mode:
        print(f"\n  [Self-Check] Verifying evaluator accuracy...")
        from test_cases import get_expected_outcomes
        expected = get_expected_outcomes()
        checks_passed = 0
        for r in results:
            q = r.get("_question", "")
            for prefix, exp in expected.items():
                if q.startswith(prefix):
                    score = r["faithfulness"]["score"]
                    if exp["expected_faithfulness"] == "high" and score >= 7:
                        checks_passed += 1
                    elif exp["expected_faithfulness"] == "low" and score <= 5:
                        checks_passed += 1
                    elif exp.get("expected_relevance") == "high" and r["relevance"]["score"] >= 7:
                        checks_passed += 1
        print(f"  Expected outcome checks passed: {checks_passed}/{len(expected)}")

    print(f"\n  Done. Total cost: ${aggregate['total_cost']:.4f}")


if __name__ == "__main__":
    main()
