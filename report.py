"""
评测报告生成
============
将 RAGEvaluator 的评估结果生成为可读报告。

支持格式：
  - console: 终端彩色文本
  - html: 独立 HTML 文件（可分享）
"""

from datetime import datetime


def _bar(score: int, width: int = 10) -> str:
    """生成评分条：8/10 ========--"""
    return "#" * score + "-" * (width - score)


def print_console_report(aggregate: dict, results: list[dict],
                         title: str = "RAG Quality Evaluation Report"):
    """
    打印控制台报告。

    Args:
        aggregate: evaluator.aggregate() 的输出
        results: evaluator.evaluate_batch() 的输出
    """
    print()
    print("=" * 58)
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 58)

    # 总览
    print(f"\n  [Summary]")
    print(f"  Test cases:       {aggregate['count']}")
    print(f"  Avg Faithfulness: {aggregate['avg_faithfulness']}/10 {_bar(round(aggregate['avg_faithfulness']))}")
    print(f"  Avg Relevance:    {aggregate['avg_relevance']}/10 {_bar(round(aggregate['avg_relevance']))}")
    print(f"  Avg Completeness: {aggregate['avg_completeness']}/10 {_bar(round(aggregate['avg_completeness']))}")
    print(f"  Avg Overall:      {aggregate['avg_overall']}/10 {_bar(round(aggregate['avg_overall']))}")
    print(f"  Pass Rate:        {aggregate['pass_rate']}% (overall >= 7)")
    print(f"  Total Cost:       ${aggregate['total_cost']:.4f}")
    print(f"  Total Time:       {aggregate['total_time_ms']}ms")

    # 最差用例
    if aggregate.get("worst_cases"):
        print(f"\n  [Worst Cases]")
        for i, w in enumerate(aggregate["worst_cases"], 1):
            print(f"  #{i} [{w['overall']}/10] {w['question']}")
            print(f"      {w['reason']}")

    # 逐条明细
    print(f"\n  [Detail]")
    for i, r in enumerate(results, 1):
        f_score = r["faithfulness"]["score"]
        r_score = r["relevance"]["score"]
        c_score = r["completeness"]["score"]
        overall = r["overall"]

        # 综合等级
        if overall >= 8:
            grade = "A"
        elif overall >= 6:
            grade = "B"
        elif overall >= 4:
            grade = "C"
        else:
            grade = "D"

        question = r.get("_question", "?")[:70]
        print(f"  [{grade}] #{i} F:{f_score} R:{r_score} C:{c_score} | O:{overall}/10 | {question}")
        if r.get("improvements") and overall < 7:
            for imp in r["improvements"][:2]:
                print(f"       -> {imp}")

    print(f"\n{'=' * 58}")


def generate_html_report(aggregate: dict, results: list[dict],
                         title: str = "RAG Quality Evaluation Report",
                         output_path: str = "report.html") -> str:
    """
    生成 HTML 报告文件。

    Returns:
        输出文件路径
    """
    def score_color(s):
        if s >= 8:
            return "#22c55e"
        elif s >= 6:
            return "#eab308"
        else:
            return "#ef4444"

    def score_bar(s):
        return f'<span style="color:{score_color(s)}">{"#" * s}{"-" * (10 - s)}</span>'

    rows_html = ""
    for i, r in enumerate(results, 1):
        overall = r["overall"]
        grade = "A" if overall >= 8 else "B" if overall >= 6 else "C" if overall >= 4 else "D"
        grade_color = score_color(overall)
        question = r.get("_question", "?")
        improvements = ""
        if r.get("improvements") and overall < 7:
            improvements = "<br>".join(f"-> {imp}" for imp in r["improvements"][:2])

        rows_html += f"""
        <tr>
            <td style="color:{grade_color};font-weight:bold">{grade}</td>
            <td>{i}</td>
            <td>{question[:80]}</td>
            <td>{score_bar(r['faithfulness']['score'])} {r['faithfulness']['score']}</td>
            <td>{score_bar(r['relevance']['score'])} {r['relevance']['score']}</td>
            <td>{score_bar(r['completeness']['score'])} {r['completeness']['score']}</td>
            <td style="color:{score_color(overall)};font-weight:bold">{overall}/10</td>
            <td style="font-size:0.85em">{improvements}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #f8fafc; color: #1e293b; }}
        h1 {{ color: #0f172a; }}
        .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
        .card {{ background: white; border-radius: 8px; padding: 16px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 120px; }}
        .card .label {{ font-size: 0.8em; color: #64748b; }}
        .card .value {{ font-size: 1.5em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th {{ background: #1e293b; color: white; padding: 10px 12px; text-align: left; font-size: 0.9em; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 0.9em; }}
        tr:hover {{ background: #f1f5f9; }}
        .footer {{ text-align: center; color: #94a3b8; margin-top: 24px; font-size: 0.85em; }}
        .worst {{ background: #fef2f2; padding: 12px; border-radius: 8px; margin: 16px 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {aggregate['count']} test cases</p>

    <div class="summary">
        <div class="card"><div class="label">Avg Overall</div><div class="value" style="color:{score_color(round(aggregate['avg_overall']))}">{aggregate['avg_overall']}/10</div></div>
        <div class="card"><div class="label">Faithfulness</div><div class="value">{aggregate['avg_faithfulness']}</div></div>
        <div class="card"><div class="label">Relevance</div><div class="value">{aggregate['avg_relevance']}</div></div>
        <div class="card"><div class="label">Completeness</div><div class="value">{aggregate['avg_completeness']}</div></div>
        <div class="card"><div class="label">Pass Rate</div><div class="value">{aggregate['pass_rate']}%</div></div>
        <div class="card"><div class="label">Cost</div><div class="value">${aggregate['total_cost']:.4f}</div></div>
    </div>

    <h2>Results</h2>
    <table>
        <tr><th>Grade</th><th>#</th><th>Question</th><th>Faithfulness</th><th>Relevance</th><th>Completeness</th><th>Overall</th><th>Improvements</th></tr>
        {rows_html}
    </table>

    <div class="footer">Generated by RAG Evaluator | LLM-as-Judge powered by DeepSeek</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
