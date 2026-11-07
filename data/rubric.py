"""面试回答的可解释、可量化评分标准（rubric 2.0）。

第 5 周改进：把"模糊的五维描述"量化为每维 5 个可观察指标（checklist），
每项 20 分，命中即得分——规则评分器逐项检测、LLM 按同一清单逐项判断，
保证两种评分路径使用同一把尺子。

分数锚点与指标命中数的对应关系：
  0 项 = 0-19   缺失/离题
  1 项 = 20-39  笼统
  2 项 = 40-59  有相关内容但依据不足
  3 项 = 60-79  基本达标
  4 项 = 80-95  有具体证据
  5 项 = 96-100 证据充分、量化并可解释取舍
"""
from __future__ import annotations

# 每项指标的满分值；维度得分 = 命中项数 × POINTS_PER_HIT（LLM 允许 10 分半档）
POINTS_PER_HIT = 20.0

# 五维量化指标清单：checklist 是规则评分器与 LLM 评分的共同依据
INDICATORS = {
    "logic": {
        "name": "逻辑性",
        "weight": 0.22,
        "criterion": "是否回应题目、说明判断依据，并把原因、行动和结果连起来。",
        "checks": [
            {"id": "L1", "name": "回应题意", "desc": "直接回应题目所问，关键词与题目相关"},
            {"id": "L2", "name": "观点明确", "desc": "有清晰的观点或结论句（我认为/我会/应该…）"},
            {"id": "L3", "name": "因果依据", "desc": "说明原因或依据（因为/由于/因此/所以/基于…）"},
            {"id": "L4", "name": "分步论证", "desc": "分点或按顺序展开论证（顺序词≥2 处）"},
            {"id": "L5", "name": "结论呼应", "desc": "结尾总结并与开头观点呼应"},
        ],
    },
    "depth": {
        "name": "深度",
        "weight": 0.24,
        "criterion": "是否提供具体背景、个人职责、关键动作、约束或可核验结果。",
        "checks": [
            {"id": "D1", "name": "量化结果", "desc": "给出数字、比例或可量化的结果"},
            {"id": "D2", "name": "个人贡献", "desc": "明确自己的职责与动作（我+负责/主导/设计…）"},
            {"id": "D3", "name": "情境背景", "desc": "交代背景、团队、时间或阶段"},
            {"id": "D4", "name": "技术业务细节", "desc": "出现具体技术、工具或业务名词"},
            {"id": "D5", "name": "约束与取舍", "desc": "说明限制、成本、风险或方案取舍"},
        ],
    },
    "structure": {
        "name": "结构",
        "weight": 0.20,
        "criterion": "是否有清楚的开头、过程和结论；行为题优先检查 STAR 四要素。",
        "checks": [
            {"id": "S1", "name": "开头总起", "desc": "开头有总起或预告（首先/我会/从…方面）"},
            {"id": "S2", "name": "分层展开", "desc": "分点、编号或顺序词展开（≥2 处）"},
            {"id": "S3", "name": "过程完整", "desc": "经历/方案的过程完整（情境—行动链条，≥2 个动作词）"},
            {"id": "S4", "name": "结果导向", "desc": "给出结果、结论或产出（结果/最终/达到/上线…）"},
            {"id": "S5", "name": "收尾总结", "desc": "结尾有总结或收束（总结/综上/因此…）"},
        ],
    },
    "expression": {
        "name": "表达",
        "weight": 0.14,
        "criterion": "是否简洁、连贯、易懂；不评价口音、外貌或人格特征。",
        "checks": [
            {"id": "E1", "name": "句段适中", "desc": "句段数量 2–6 个"},
            {"id": "E2", "name": "篇幅适中", "desc": "总长度 80–800 字"},
            {"id": "E3", "name": "无冗余重复", "desc": "无明显口头禅式的连续重复"},
            {"id": "E4", "name": "用词具体", "desc": "使用具体动作词（≥2 个不同：负责/推动/设计…）"},
            {"id": "E5", "name": "衔接自然", "desc": "连接词/顺序词使表述连贯（≥2 处）"},
        ],
    },
    "accuracy": {
        "name": "准确度",
        "weight": 0.20,
        "criterion": "只检查回答与题意及用户提供资料的一致性；没有证据时不得臆测事实。",
        "checks": [
            {"id": "A1", "name": "紧扣题意", "desc": "内容围绕题目，无离题扩展"},
            {"id": "A2", "name": "与资料一致", "desc": "陈述与候选人提供的资料核对一致（有资料时）"},
            {"id": "A3", "name": "数据可核验", "desc": "数字/事实有出处，或与资料中的数字一致"},
            {"id": "A4", "name": "无过度断言", "desc": "没有绝对化、无法证实的断言（保证/百分百…）"},
            {"id": "A5", "name": "标注不确定性", "desc": "信息不足时说明不确定或补充条件"},
        ],
    },
}

# 供旧调用方使用的精简版（name/weight/criterion），保持向后兼容
DIMENSIONS = {
    key: {k2: v2 for k2, v2 in value.items() if k2 != "checks"}
    for key, value in INDICATORS.items()
}

SCORE_ANCHORS = (
    "0-39：缺失、离题或无法从回答中找到相应证据；"
    "40-59：有相关内容但笼统，关键依据或步骤不清；"
    "60-74：基本达标，能回答问题并给出部分可追踪细节；"
    "75-89：表现扎实，论据具体、推理连贯，能说明个人贡献或结果；"
    "90-100：证据充分且高度贴题，能量化结果、说明取舍并验证结论。"
)


def indicator_items(dim: str) -> list[dict]:
    """某维度的指标清单（id/name/desc），未知维度返回空列表。"""
    return list(INDICATORS.get(dim, {}).get("checks", []))


def dimension_score(hits: int, partial: int = 0) -> float:
    """按命中数换算维度得分：每项 20 分，半档 10 分，上限 100。"""
    return round(min(100.0, hits * POINTS_PER_HIT + partial * 10.0), 1)


def weighted_score(scores: dict[str, float]) -> float:
    """按维度权重计算内容总分（0-100）。"""
    return round(
        sum(
            float(scores[key]) * INDICATORS[key]["weight"]
            for key in INDICATORS
            if key in scores
        ),
        1,
    )


def checklist_text() -> str:
    """把量化指标清单渲染为提示词用的纯文本。"""
    lines = []
    for dim, meta in INDICATORS.items():
        lines.append(
            f"{meta['name']}（{meta['weight']:.0%}，每项 {POINTS_PER_HIT:.0f} 分）："
            f"{meta['criterion']}"
        )
        for item in meta["checks"]:
            lines.append(f"  {item['id']} {item['name']}：{item['desc']}")
    return "\n".join(lines)
