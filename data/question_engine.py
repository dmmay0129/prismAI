"""个性化出题引擎：根据岗位、行业、级别、面试类型、关注点和 JD 生成题目。

当前使用模板 + 用户画像组合，离线可用；如果后续接入 Ollama，可在
generate_questions 内部增加 LLM 分支，失败后仍回退到本模板引擎。
"""
from __future__ import annotations

from typing import Any, Dict, List

INTERVIEW_TYPES = ["综合面", "技术面", "行为面", "压力面", "HR面"]
LEVELS = ["校招/实习", "初级", "中级", "高级", "管理"]
DIFFICULTIES = ["基础", "进阶", "挑战"]
FOCUS_OPTIONS = [
    "项目经历",
    "算法与编码",
    "系统设计",
    "团队协作",
    "领导力",
    "稳定性与动机",
    "业务理解",
    "压力应对",
    "沟通表达",
    "综合能力",
]

_FOCUS_CATEGORY = {
    "项目经历": "项目深挖",
    "算法与编码": "技术类",
    "系统设计": "技术类",
    "团队协作": "管理协作",
    "领导力": "管理协作",
    "稳定性与动机": "动机匹配",
    "业务理解": "岗位业务",
    "压力应对": "压力追问",
    "沟通表达": "行为类",
}

_DEFAULT_ORDER = [
    "行为类", "技术类", "情景类", "项目深挖",
    "岗位业务", "压力追问", "管理协作", "动机匹配",
]

_INTERVIEW_ORDERS = {
    "综合面": _DEFAULT_ORDER,
    "技术面": [
        "技术类", "项目深挖", "岗位业务", "情景类",
        "压力追问", "行为类", "管理协作", "动机匹配",
    ],
    "行为面": [
        "行为类", "情景类", "项目深挖", "管理协作",
        "动机匹配", "压力追问", "技术类", "岗位业务",
    ],
    "压力面": [
        "压力追问", "情景类", "行为类", "项目深挖",
        "技术类", "管理协作", "岗位业务", "动机匹配",
    ],
    "HR面": [
        "动机匹配", "行为类", "管理协作", "情景类",
        "项目深挖", "压力追问", "技术类", "岗位业务",
    ],
}

_FOLLOW_UPS = {
    "行为类": "如果重来一次，你会改变哪个关键动作？",
    "技术类": "这个方案最大的瓶颈是什么？你会如何验证？",
    "情景类": "如果资源再减少一半，你的方案会怎么调整？",
    "项目深挖": "你个人贡献和团队成果如何区分？",
    "岗位业务": "你会用什么数据证明这个判断？",
    "压力追问": "请举一个反例，说明你的方案可能不成立。",
    "管理协作": "如果对方仍然不配合，你的下一步是什么？",
    "动机匹配": "如果现实与预期不符，你会如何调整？",
}

# (问题模板, 说明性标签)
_TEMPLATES: Dict[str, List[str]] = {
    "行为类": [
        "请结合你在{industry}的经历，介绍一次你在{role}相关工作中主动承担额外责任的例子。",
        "描述一次你和团队成员意见不一致的经历，你当时如何推动达成共识？",
        "请分享一次你失败或未达预期的经历，你从中复盘出了什么？",
        "在{level}阶段，你如何安排优先级并保证关键任务按时交付？",
        "请举例说明你如何向非专业同事解释一个复杂问题，并推动对方行动。",
    ],
    "技术类": [
        "针对{role}，请介绍你最熟悉的一项核心技术，并说明它在真实项目中的取舍。",
        "如果让你从零设计一个支撑{industry}业务的高并发模块，你会如何划分边界并做容量评估？",
        "请说明一次你定位线上性能问题的过程：如何发现、如何验证、最终如何修复？",
        "在{role}工作中，你如何保证代码质量和可维护性？请结合具体实践说明。",
        "请对比两种你熟悉的与{focus_text}相关的方案，说明不同业务场景下的选择依据。",
    ],
    "情景类": [
        "上线前一天发现严重缺陷，但业务方坚持按时发布，你会如何处理？",
        "如果核心需求在截止日期前无法完成，你会如何与{stakeholder}沟通并给出方案？",
        "当你接手一个历史包袱很重的项目时，你会如何在前两周梳理风险并制定计划？",
        "如果线上故障影响到{industry}客户，你会如何组织排查、止损和复盘？",
        "如果团队同时接到多个高优需求，你会如何排序并说明理由？",
    ],
    "项目深挖": [
        "请选一个你最有代表性的{role}项目，说明背景、你的职责、关键决策和最终结果。",
        "在这个项目中，最大的技术风险是什么？你是如何提前识别和降低风险的？",
        "如果重新做一次，你会在架构或流程上改变什么？为什么？",
        "请说明你在项目中如何衡量成功，最终的量化结果是什么？",
        "项目里有没有你做了但最终被推翻的方案？你如何复盘？",
    ],
    "岗位业务": [
        "结合你对{industry}的理解，你认为{role}最关键的三项能力是什么？",
        "如果让你在一个月内熟悉新的业务领域，你会如何制定学习计划？",
        "针对 JD 中的“{jd_snippet}”，请说明你的理解和相关实践。",
        "你如何看待{role}在团队中的价值？请结合业务目标说明。",
        "如果你发现业务指标和用户体验存在冲突，你会如何权衡？",
    ],
    "压力追问": [
        "你刚才提到的方案听起来成本很高，为什么不用更简单的方案？",
        "如果领导否定了你的方案，但你认为它是对的，你会怎么做？",
        "如果必须在信息不完整的情况下做决定，你会如何降低风险？",
        "请举一个你被质疑或被挑战的例子，你当时如何回应？",
        "如果这个项目失败了，你认为最可能的原因是什么？",
    ],
    "管理协作": [
        "请说明你如何协调多个{stakeholder}，推动一个跨团队目标落地。",
        "当团队成员绩效不达预期时，你会如何反馈和辅导？",
        "你如何做技术或方案评审，既保证质量又不拖慢进度？",
        "请分享一次你处理团队冲突的经历。",
        "在{level}阶段，你如何培养团队或其他成员？",
    ],
    "动机匹配": [
        "你为什么选择{industry}和{role}？请结合你的经历说明。",
        "你未来 1-3 年的职业规划是什么？这个岗位如何匹配？",
        "你如何判断一家公司或团队是否适合自己？",
        "在工作中什么最让你有成就感？什么最让你消耗？",
        "如果同时拿到两个 offer，你会如何做决策？",
    ],
}


def _build_context(profile: Dict[str, Any]) -> Dict[str, Any]:
    role = str(profile.get("role") or "目标岗位").strip()
    industry = str(profile.get("industry") or "目标行业").strip()
    level = str(profile.get("level") or "当前").strip()
    interview_type = str(profile.get("interview_type") or "综合面").strip()
    if interview_type not in INTERVIEW_TYPES:
        interview_type = "综合面"
    difficulty = str(profile.get("difficulty") or "进阶").strip()
    if difficulty not in DIFFICULTIES:
        difficulty = "进阶"

    raw_focus = profile.get("focus") or []
    if isinstance(raw_focus, str):
        raw_focus = [raw_focus]
    focus = [f for f in raw_focus if f in FOCUS_OPTIONS] or ["综合能力"]

    jd = " ".join(str(profile.get("jd") or "").split())
    jd_snippet = (jd[:70] + "…") if len(jd) > 70 else (jd or "岗位要求中的关键能力")

    return {
        "role": role,
        "industry": industry,
        "level": level,
        "interview_type": interview_type,
        "difficulty": difficulty,
        "focus": focus,
        "focus_text": "、".join(focus),
        "jd_snippet": jd_snippet,
        "stakeholder": "业务方或跨团队同事",
        "skill": focus[0] if focus and focus[0] != "综合能力" else role,
    }


def _category_order(interview_type: str, focus: List[str], has_jd: bool) -> List[str]:
    order: List[str] = []
    for item in focus:
        category = _FOCUS_CATEGORY.get(item)
        if category and category not in order:
            order.append(category)
    if has_jd:
        order = ["岗位业务"] + [c for c in order if c != "岗位业务"]
    for category in _INTERVIEW_ORDERS.get(interview_type, _DEFAULT_ORDER):
        if category not in order:
            order.append(category)
    return order


def generate_questions(profile: Dict[str, Any], count: int = 6) -> List[dict]:
    """根据用户画像生成去重后的个性化题目。

    profile 支持：role / industry / level / interview_type /
    difficulty / focus(list) / jd。
    """
    ctx = _build_context(profile)
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 6
    count = max(1, min(12, count))

    order = _category_order(
        ctx["interview_type"], ctx["focus"], bool(profile.get("jd"))
    )
    selected: List[dict] = []
    seen = set()
    cursor = {category: 0 for category in order}

    while len(selected) < count:
        progressed = False
        for category in order:
            if len(selected) >= count:
                break
            templates = _TEMPLATES.get(category) or []
            index = cursor.get(category, 0)
            if index >= len(templates):
                continue
            cursor[category] = index + 1
            content = templates[index].format(**ctx)
            if content in seen:
                continue
            seen.add(content)
            selected.append(
                {
                    "id": f"PF{len(selected) + 1:03d}",
                    "category": category,
                    "content": content,
                    "follow_up": _FOLLOW_UPS.get(category, ""),
                    "difficulty": ctx["difficulty"],
                    "tags": [
                        "个性化",
                        ctx["difficulty"],
                        ctx["interview_type"],
                    ]
                    + [f for f in ctx["focus"] if f != "综合能力"],
                    "source": "模板生成",
                }
            )
            progressed = True
        if not progressed:
            break

    return selected

