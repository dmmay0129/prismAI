"""PrismAI 多模态 AI 面试教练 —— Streamlit 表现层（MVP 入口）。

运行：streamlit run app.py
表现层不做任何 AI 分析，仅采集输入并渲染门面返回的结构化报告（第4周五.1约束）。
视觉风格：浅鼠尾草绿主题，白色卡片 + 绿色细边。
"""
from __future__ import annotations

import html
import os
import sys
import tempfile
import time
import uuid

# Streamlit 热重载只重新执行主脚本；已导入的本地模块会残留在 sys.modules，
# 修改 core/data/analyzers/ui 等目录下的文件不会生效。每次 run 前强制失效
# 项目内模块（排除 .venv 第三方包），保证 import 时始终读取磁盘最新代码。
# 强制重载会丢掉模块级缓存（如 Whisper 模型，加载成本高）：
# 失效前先暂存，等本轮 import 完成后归还给新模块。
_PRESERVE_ATTRS = {"analyzers.speech_analyzer": "_MODEL_CACHE"}
_preserved = {}
for _mod_name, _attr in _PRESERVE_ATTRS.items():
    _old = sys.modules.get(_mod_name)
    if _old is not None:
        _cache = getattr(_old, _attr, None)
        if _cache:
            _preserved[_mod_name] = (_attr, dict(_cache))
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for _m, _mod in list(sys.modules.items()):
    _file = getattr(_mod, "__file__", "") or ""
    if (
        not _file
        or _m == "__main__"
        or ".venv" in _file
        or "site-packages" in _file
    ):
        continue
    try:
        if os.path.abspath(_file).startswith(_PROJECT_ROOT):
            sys.modules.pop(_m, None)
    except (ValueError, OSError):
        continue

import streamlit as st

from core.diagnostics import check_environment
from core.errors import AnalysisBusyError
from core.facade import InterviewCoachFacade
from core.input_resolver import InputValidationError
from core.settings import get_settings
from data.question_engine import (
    DIFFICULTIES,
    FOCUS_OPTIONS,
    INTERVIEW_TYPES,
    LEVELS,
    generate_questions,
)
from data.interview_engine import generate_follow_up
from data.jd_profiles import PROFILE_NAMES, profile_jd_text
from data.prompts import (
    closing_text,
    feedback_voice_text,
    opening_text,
    review_summary,
    scoring_prompt,
)
from data.questions import CATEGORIES, QUESTIONS, by_category
from data.storage import (
    list_interview_sessions,
    load_history,
    load_interview_turns,
    load_knowledge,
    load_report,
    save_interview_turn,
    save_knowledge,
)
from ui.call_component import broadcast, decode_wav_b64, emit_voice, parse_event, render_call
from ui.login import render_login

# 归还上一轮暂存的模块级缓存（见文件顶部失效逻辑）
for _mod_name, (_attr, _cache) in _preserved.items():
    _target = sys.modules.get(_mod_name)
    if _target is not None:
        getattr(_target, _attr, {}).update(_cache)

# 通话/多轮面试的业务状态键（不含控件 widget 键，退出登录与超时时清理）
_CALL_STATE_KEYS = (
    "call_user", "call_session_id", "call_turns", "call_question",
    "call_limit", "call_persist", "call_active", "call_started",
    "call_opening_done", "call_notice", "call_last_sig",
    "call_pending_broadcast", "_call_bcast_seq",
)
_DIALOG_STATE_KEYS = (
    "dialog_user", "dialog_session_id", "dialog_turns", "dialog_question",
    "dialog_active", "dialog_follow_up_source", "dialog_opening",
    "dialog_opening_done",
)


@st.cache_data(ttl=30, show_spinner=False)
def _system_diagnostics():
    return check_environment()

# ---------------------------------------------------------------------------
# 全局样式
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PrismAI 面试教练", page_icon="🎯", layout="wide")

CSS = """
<style>
/* 基础 */
:root {
  --bg:        #eef2ee;
  --card:      #ffffff;
  --green:     #5a9b7e;
  --green-2:   #4a8c6e;
  --green-bg:  #e8f3ee;
  --green-line:#b7d4c4;
  --red:       #d6606b;
  --text:      #2d3436;
  --muted:     #6b7280;
  --border:    #e0e8e0;
  --radius:    12px;
}
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stSidebar"]          { background: #ffffff; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] *        { color: var(--text); }
h1, h2, h3, h4 { color: var(--text) !important; font-family: "PingFang SC","Microsoft YaHei",sans-serif; }

/* 顶部主标题 */
.hero-title {
  font-size: 2.2rem; font-weight: 700; color: var(--text);
  text-align: center; margin: 0.4rem 0 0.1rem; letter-spacing: 1px;
}
.hero-sub {
  text-align: center; color: var(--muted); font-size: 0.95rem; margin-bottom: 1.2rem;
}
.hero-underline {
  width: 90px; height: 4px; margin: 0 auto 1.6rem;
  background: linear-gradient(90deg, var(--green) 0%, var(--green-2) 100%);
  border-radius: 2px;
}

/* 卡片通用 */
.prism-card {
  background: var(--card); border-radius: var(--radius);
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  border: 1px solid var(--border);
  padding: 1.1rem 1.3rem; margin-bottom: 1rem;
}
.card-title { font-size: 1.15rem; font-weight: 700; color: var(--text); margin-bottom: 0.6rem; }
.card-sub   { font-size: 0.85rem; color: var(--muted); line-height: 1.55; }

/* 左侧功能卡 */
.feature-card {
  background: var(--card); border-radius: var(--radius);
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  border: 1px solid var(--border); border-bottom: 3px solid var(--green);
  padding: 1rem 1.1rem; margin-bottom: 0.9rem; transition: transform .15s;
}
.feature-card:hover { transform: translateY(-2px); }
.feature-card .ft-title { font-size: 1.15rem; font-weight: 700; color: var(--text); }
.feature-card .ft-desc  { font-size: 0.82rem; color: var(--muted); margin-top: 0.3rem; line-height:1.5; }
.user-card { border-bottom-color: var(--red) !important; }
.user-card .ft-desc { color: var(--text); font-weight: 600; }

/* 题目展示卡 */
.question-card {
  background: var(--green-bg); border-left: 4px solid var(--green);
  border-radius: var(--radius); padding: 1rem 1.2rem; margin: 0.6rem 0 1rem;
}
.question-card .q-tag {
  display: inline-block; background: var(--green); color: #fff;
  font-size: 0.72rem; padding: 2px 10px; border-radius: 10px; margin-right: 8px;
  vertical-align: middle;
}
.question-card .q-text { font-size: 1rem; color: var(--text); font-weight: 500; line-height:1.6; vertical-align: middle; }
.question-card .q-index {
  float: right; font-size: 0.78rem; color: var(--muted);
  padding-top: 4px;
}

/* 题目选择 radio 菜单 */
[data-testid="stRadio"] {
  width: 100% !important;
}
[data-testid="stRadio"] > div {
  background: var(--card);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  padding: 0.5rem;
  max-height: 260px;
  overflow-y: auto;
  width: 100% !important;
}
[data-testid="stRadio"] [role="radiogroup"] {
  width: 100% !important;
}
[data-testid="stRadio"] label {
  padding: 0.7rem 0.9rem !important;
  border-radius: 8px !important;
  margin-bottom: 4px !important;
  transition: background .15s;
  cursor: pointer;
}
[data-testid="stRadio"] label:hover {
  background: var(--green-bg) !important;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: var(--green-bg) !important;
  border-left: 3px solid var(--green);
}
[data-testid="stRadio"] label span {
  color: var(--text) !important;
  font-size: 0.92rem !important;
}

/* 评分卡 */
.score-big {
  font-size: 2.6rem; font-weight: 800; color: var(--green-2);
  line-height: 1; letter-spacing: -1px;
}
.score-label { font-size: 0.85rem; color: var(--muted); margin-top: 0.3rem; }
.dim-row { display:flex; justify-content:space-between; align-items:center;
           padding: 0.55rem 0; border-bottom: 1px solid var(--border); }
.dim-row:last-child { border-bottom:none; }
.dim-name { font-size: 0.92rem; color: var(--text); font-weight: 500; }
.dim-val  { font-size: 1rem;  font-weight: 700; color: var(--green-2); }
.badge    { font-size: 0.7rem; padding: 1px 8px; border-radius: 8px; margin-left:6px; }
.badge-ok  { background: var(--green-bg); color: var(--green-2); }
.badge-deg { background: #fdf0e6; color: #c97a2b; }

/* 建议 */
.sug-item {
  background: var(--green-bg); border-radius: 8px;
  padding: 0.6rem 0.9rem; margin-bottom: 0.5rem;
  font-size: 0.9rem; color: var(--text); line-height:1.5;
  border-left: 3px solid var(--green);
}

/* 上传区美化 */
[data-testid="stFileUploader"] { background: var(--card); border-radius: var(--radius);
  border: 1.5px dashed var(--green-line); padding: 0.2rem; }
[data-testid="stFileUploader"]:hover { border-color: var(--green); }

/* 按钮 */
.stButton > button {
  background: var(--green) !important; color: #fff !important;
  border: none !important; border-radius: 8px !important;
  font-weight: 600 !important; padding: 0.5rem 1.6rem !important;
  transition: background .2s;
}
.stButton > button:hover { background: var(--green-2) !important; }
.stButton > button:disabled { background: #c8d8d0 !important; color: #8aa098 !important; }

/* selectbox 容器 */
[data-testid="stSelectbox"] > div > div {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 8px;
}

/* spinner 文字 */
.stSpinner > div { color: var(--green-2); }

/* 主内容区加宽：让 stMain 占满侧边栏以外的全部空间 */
[data-testid="stMain"] {
  flex: 1 1 0% !important;
  width: 100% !important;
  max-width: none !important;
}
[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 100% !important;
  width: 100% !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}
/* 题目选择区按钮卡片 */
.question-picker button {
  height: auto !important;
  min-height: 64px !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 0.8rem 1rem !important;
  border-radius: 10px !important;
  font-size: 0.92rem !important;
  line-height: 1.45 !important;
  white-space: normal !important;
}
.question-picker button[data-testid="stBaseButton-secondary"] {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1.5px solid var(--border) !important;
}
.question-picker button[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--green) !important;
  background: var(--green-bg) !important;
}
.question-picker button[data-testid="stBaseButton-primary"] {
  background: var(--green) !important;
  color: #fff !important;
  border: 1.5px solid var(--green) !important;
}
.cat-picker button {
  border-radius: 20px !important;
  font-size: 0.85rem !important;
  height: 36px !important;
}
.cat-picker button[data-testid="stBaseButton-secondary"] {
  background: var(--card) !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
}
.cat-picker button[data-testid="stBaseButton-primary"] {
  background: var(--green) !important;
  color: #fff !important;
  border: 1px solid var(--green) !important;
}

/* 标题下划线（参考图的绿色短下划线） */
.section-head {
  font-size: 1.3rem; font-weight: 700; color: var(--text);
  padding-bottom: 0.4rem; margin-bottom: 1rem;
  border-bottom: 2px solid var(--green);
  display: inline-block;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 顶部标题
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">PrismAI 多模态 AI 面试教练</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">听见表达 · 看见状态 · 评出水平</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-underline"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 登录校验：未登录时只渲染登录/注册页
# ---------------------------------------------------------------------------
auth_user = st.session_state.get("auth_user")
if auth_user:
    idle_timeout = int(get_settings()["auth"]["idle_timeout_sec"])
    last_active = float(st.session_state.get("auth_last_active", 0) or 0)
    if idle_timeout and last_active and time.time() - last_active > idle_timeout:
        for key in (
            "auth_user", "auth_last_active", "report", "report_key",
            "history_report", "history_report_sid", "history_report_missing",
        ):
            st.session_state.pop(key, None)
        for key in _CALL_STATE_KEYS + _DIALOG_STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
    st.session_state["auth_last_active"] = time.time()
if not auth_user:
    render_login()

# ---------------------------------------------------------------------------
# 左侧功能导航（仿参考图左侧卡片）
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="feature-card user-card">'
        f'<div class="ft-title">当前用户</div>'
        f'<div class="ft-desc">{html.escape(str(auth_user))}</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("退出登录", key="logout", use_container_width=True):
        for key in (
            "auth_user", "auth_last_active", "report", "report_key",
            "history_report", "history_report_sid", "history_report_missing",
        ):
            st.session_state.pop(key, None)
        for key in _CALL_STATE_KEYS + _DIALOG_STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    st.markdown('<div class="feature-card"><div class="ft-title">实时面试提醒</div>'
                '<div class="ft-desc">语速、填充词实时预警，帮你把握作答节奏</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="feature-card"><div class="ft-title">AI 模拟面试</div>'
                '<div class="ft-desc">模拟真实面试场景，多维度深度评估</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="feature-card"><div class="ft-title">一键 AI 评估</div>'
                '<div class="ft-desc">内容 / 语音 / 视觉三路融合，输出量化报告</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="feature-card"><div class="ft-title">深度面试复盘</div>'
                '<div class="ft-desc">五维雷达图 + 具体建议，精准定位提升方向</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="feature-card"><div class="ft-title">进步趋势追踪</div>'
                '<div class="ft-desc">历史练习分数曲线，见证每一次成长</div></div>',
                unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    with st.expander("系统状态"):
        for item in _system_diagnostics():
            icon = {"ok": "✅", "warn": "⚠️", "error": "❌"}.get(
                item.get("status"), "•"
            )
            st.markdown(
                f"{icon} **{item.get('name', '')}**：{item.get('detail', '')}"
            )

# ---------------------------------------------------------------------------
# 主体：面试发起卡（面试设置移至主页面）
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head">开始模拟面试</div>', unsafe_allow_html=True)

knowledge = load_knowledge(str(auth_user)) if auth_user else {}
knowledge_prefix = f"knowledge_{auth_user}"
with st.expander("我的面试知识库与 AI 提示词", expanded=False):
    st.caption(
        "资料按当前账号保存在本机 SQLite 中，仅用于生成追问和核对回答；"
        "面试官风格不会改变固定评分标准。"
    )
    with st.form(f"knowledge_form_{auth_user}"):
        candidate_profile = st.text_area(
            "个人背景与核心技能",
            value=knowledge.get("candidate_profile", ""),
            key=f"{knowledge_prefix}_profile",
            height=90,
            placeholder="填写经历、技能、职责范围；避免填写身份证号、密码等敏感信息。",
        )
        project_evidence = st.text_area(
            "项目事实 / 可核验成果",
            value=knowledge.get("project_evidence", ""),
            key=f"{knowledge_prefix}_projects",
            height=110,
            placeholder="按项目记录：目标、你的职责、采取的行动、数据结果与证据来源。",
        )
        jd_ver = st.session_state.get(f"{knowledge_prefix}_jd_ver", 0)
        jd_draft = st.session_state.get(f"{knowledge_prefix}_jd_draft")
        job_description = st.text_area(
            "目标岗位 JD / 面试要求",
            value=(jd_draft if jd_draft is not None
                   else knowledge.get("job_description", "")),
            key=f"{knowledge_prefix}_jd_v{jd_ver}",
            height=100,
        )
        profile_row = st.columns([2, 1])
        with profile_row[0]:
            jd_profile = st.selectbox(
                "从内置岗位画像导入 JD",
                ["— 选择岗位 —"] + PROFILE_NAMES,
                key=f"{knowledge_prefix}_jd_profile",
                help="内置 10 个常见岗位的职责、技能与 JD 模板；导入后可再修改。",
            )
        with profile_row[1]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            import_jd = st.form_submit_button("导入到 JD")
        interviewer_style = st.text_input(
            "面试官风格（可选）",
            value=knowledge.get("interviewer_style", ""),
            key=f"{knowledge_prefix}_style",
            max_chars=160,
            placeholder="例如：友善但会追问数据和技术取舍",
        )
        review_ver = st.session_state.get(f"{knowledge_prefix}_review_ver", 0)
        review_draft = st.session_state.get(f"{knowledge_prefix}_review_draft")
        interview_review = st.text_area(
            "面试复盘沉淀（每场多轮面试结束后自动追加，可修改）",
            value=(review_draft if review_draft is not None
                   else knowledge.get("interview_review", "")),
            key=f"{knowledge_prefix}_review_v{review_ver}",
            height=120,
            placeholder="完成多轮面试后自动沉淀：每轮得分与最需提升的点。",
        )
        knowledge_saved = st.form_submit_button("保存知识库")
    if import_jd and jd_profile and jd_profile != "— 选择岗位 —":
        st.session_state[f"{knowledge_prefix}_jd_draft"] = profile_jd_text(jd_profile)
        st.session_state[f"{knowledge_prefix}_jd_ver"] = (
            st.session_state.get(f"{knowledge_prefix}_jd_ver", 0) + 1
        )
        st.rerun()
    if knowledge_saved:
        try:
            save_knowledge(str(auth_user), {
                "candidate_profile": candidate_profile,
                "project_evidence": project_evidence,
                "job_description": job_description,
                "interviewer_style": interviewer_style,
                "interview_review": interview_review,
            })
            knowledge = load_knowledge(str(auth_user))
            st.session_state[f"{knowledge_prefix}_jd_draft"] = None
            st.session_state[f"{knowledge_prefix}_review_draft"] = None
            st.success("知识库已保存，后续单题评估和模拟面试会使用这些资料。")
        except (OSError, ValueError) as exc:
            st.error(f"保存知识库失败：{exc}")

with st.expander("评分量表与 AI 评估提示词"):
    st.markdown(
        "**总分 = 内容质量 × 55% + 语音表达 × 45%。** "
        "画面可见度单独反馈，不影响能力总分。"
    )
    st.markdown(
        "**量化评分规则（rubric 2.0）：** 内容五维各设 5 个可观察指标，"
        "每项 20 分，命中即得分。规则评分器与 LLM 使用同一指标清单，"
        "保证两条评分路径共用一把尺子。"
    )
    st.markdown(
        "- 逻辑性 22%：回应题意 / 观点明确 / 因果依据 / 分步论证 / 结论呼应\n"
        "- 深度 24%：量化结果 / 个人贡献 / 情境背景 / 技术业务细节 / 约束取舍\n"
        "- 结构 20%：开头总起 / 分层展开 / 过程完整 / 结果导向 / 收尾总结\n"
        "- 表达 14%：句段适中 / 篇幅适中 / 无冗余重复 / 用词具体 / 衔接自然\n"
        "- 准确度 20%：紧扣题意 / 与资料一致 / 数据可核验 / 无过度断言 / 标注不确定性"
    )
    st.caption(
        "分档锚点：命中 0 项缺失/离题；1 项笼统；2 项有内容但依据不足；"
        "3 项基本达标；4 项有具体证据；5 项证据充分并能解释取舍。"
        "准确度无可核验资料时固定 50 分，不臆测事实。"
    )
    st.code(scoring_prompt(), language="text")

with st.container():
    # 状态初始化（只执行一次，避免重跑覆盖用户选择）
    if "active_cat" not in st.session_state:
        st.session_state.active_cat = CATEGORIES[0]
    if "selected_qid" not in st.session_state:
        st.session_state.selected_qid = by_category(CATEGORIES[0])[0]["id"]

    with st.expander("个性化出题（按岗位 / JD 生成）", expanded=False):
        row1 = st.columns(2)
        with row1[0]:
            role = st.text_input(
                "目标岗位", key="pq_role", placeholder="例如：后端工程师"
            )
        with row1[1]:
            industry = st.text_input(
                "目标行业", key="pq_industry", placeholder="例如：互联网 / 金融"
            )
        row2 = st.columns(2)
        with row2[0]:
            level = st.selectbox("职级 / 经验", LEVELS, index=2, key="pq_level")
        with row2[1]:
            interview_type = st.selectbox(
                "面试类型", INTERVIEW_TYPES, key="pq_type"
            )
        row3 = st.columns(2)
        with row3[0]:
            difficulty = st.selectbox(
                "难度", DIFFICULTIES, index=1, key="pq_difficulty"
            )
        with row3[1]:
            question_count = st.slider("题目数量", 5, 10, 6, key="pq_count")
        focus = st.multiselect(
            "重点关注", FOCUS_OPTIONS,
            default=["项目经历", "沟通表达"], key="pq_focus",
        )
        jd = st.text_area(
            "岗位 JD / 面试要求（可选）", key="pq_jd", height=100,
            placeholder="粘贴岗位职责/要求；生成的题目会引用其中的关键点。",
        )
        if st.button(
            "生成个性化题目", type="primary",
            key="pq_generate", use_container_width=True,
        ):
            profile = {
                "role": role.strip(),
                "industry": industry.strip(),
                "level": level,
                "interview_type": interview_type,
                "difficulty": difficulty,
                "focus": focus,
                "jd": jd.strip(),
            }
            generated = generate_questions(profile, question_count)
            if generated:
                st.session_state["custom_questions"] = generated
                st.session_state["custom_profile"] = profile
                st.session_state["custom_version"] = (
                    st.session_state.get("custom_version", 0) + 1
                )
                st.session_state["selected_qid"] = generated[0]["id"]
                st.session_state.pop("report", None)
                st.session_state.pop("report_key", None)
                st.rerun()
            else:
                st.error("未能生成题目，请调整条件后重试。")

        custom = st.session_state.get("custom_questions") or []
        if custom:
            st.toggle(
                "使用已生成的个性化题目",
                value=True,
                key=f"pq_use_custom_{st.session_state.get('custom_version', 0)}",
            )
            profile = st.session_state.get("custom_profile", {})
            st.caption(
                f"已生成 {len(custom)} 道题｜"
                f"目标：{profile.get('role') or '目标岗位'}｜"
                f"类型：{profile.get('interview_type') or '综合面'}｜"
                f"难度：{profile.get('difficulty') or '进阶'}"
            )
            if st.button(
                "清除个性化题目", key="pq_clear", use_container_width=True
            ):
                st.session_state["custom_questions"] = []
                st.session_state.pop("custom_profile", None)
                st.session_state["selected_qid"] = by_category(
                    st.session_state.active_cat
                )[0]["id"]
                st.session_state.pop("report", None)
                st.session_state.pop("report_key", None)
                st.rerun()

    custom_questions = st.session_state.get("custom_questions") or []
    custom_version = st.session_state.get("custom_version", 0)
    use_custom = bool(custom_questions) and bool(
        st.session_state.get(f"pq_use_custom_{custom_version}", True)
    )

    if use_custom:
        q_list = custom_questions
        q_ids = [q["id"] for q in q_list]
        cur_idx = (
            q_ids.index(st.session_state.selected_qid)
            if st.session_state.selected_qid in q_ids else 0
        )
        question = st.radio(
            "选择个性化题目",
            options=q_list,
            index=cur_idx,
            format_func=lambda q: (
                f"[{q['id']}] {q['content']}"
                + (f"（{q.get('difficulty', '')}）" if q.get("difficulty") else "")
            ),
            key=f"custom_q_{st.session_state.get('custom_version', 0)}",
            label_visibility="collapsed",
        )
        st.session_state.selected_qid = question["id"]
        category = question.get("category", "个性化")
        if question.get("follow_up"):
            st.caption(f"追问：{question['follow_up']}")
        if question.get("tags"):
            st.caption("标签：" + " · ".join(str(t) for t in question["tags"]))
    else:
        # 类别切换按钮（每行 4 个）
        st.markdown('<div class="cat-picker">', unsafe_allow_html=True)
        for start in range(0, len(CATEGORIES), 4):
            row_cats = CATEGORIES[start:start + 4]
            cat_cols = st.columns(len(row_cats))
            for index, cat in enumerate(row_cats):
                with cat_cols[index]:
                    active = st.session_state.active_cat == cat
                    if st.button(
                        cat, key=f"cat_{cat}", use_container_width=True,
                        type="primary" if active else "secondary",
                    ):
                        st.session_state.active_cat = cat
                        st.session_state.selected_qid = by_category(cat)[0]["id"]
        st.markdown('</div>', unsafe_allow_html=True)

        q_list = by_category(st.session_state.active_cat)
        q_ids = [q["id"] for q in q_list]
        cur_idx = (
            q_ids.index(st.session_state.selected_qid)
            if st.session_state.selected_qid in q_ids else 0
        )
        question = st.radio(
            "选择题目",
            options=q_list,
            index=cur_idx,
            format_func=lambda q: f"[{q['id']}] {q['content']}",
            key=f"q_radio_{st.session_state.selected_qid}",
            label_visibility="collapsed",
        )
        st.session_state.selected_qid = question["id"]
        category = st.session_state.active_cat

    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.file_uploader(
            "上传回答音频（WAV，可选）", type=["wav"],
            help="音频和视频至少上传一个；只上传视频时会自动提取音轨。",
        )
    with col2:
        video_file = st.file_uploader(
            "上传回答视频（MP4/MOV，可选）", type=["mp4", "mov"],
            help="视频用于视觉分析；若没有单独音频，会自动从视频提取音轨。",
        )
    st.caption("提示：音频与视频至少上传一个；只上传视频也能开始评估。")


def _save_temp(uploaded, default_suffix: str) -> str:
    suffix = os.path.splitext(getattr(uploaded, "name", ""))[1] or default_suffix
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.read())
    return path


def _emit_speech(texts) -> None:
    """在主文档注入脚本，依次朗读多段文本（AI 面试官播报）。

    优先播放服务端合成的微软神经语音 MP3（音色更真实），
    合成失败时自动退回浏览器 TTS；实现见 ui/call_component.emit_voice。
    """
    emit_voice(texts)


def _voice(key: str, texts, auto: bool = False) -> None:
    """AI 面试官语音播报组件：自动尝试朗读 + 🔊 按钮兜底。

    浏览器自动播放策略要求近期用户操作：用户点击「开始/提交」后的
    页面重跑会尝试自动朗读；被拦截时点击按钮即可（按钮点击本身
    就是用户操作，一定可播）。
    """
    if not texts:
        return
    texts = [str(t) for t in texts if str(t).strip()]
    if not texts:
        return
    st.session_state.setdefault("_voice_queue", [])
    if st.button(
        "🔊 播放语音", key=f"voice_btn_{key}",
        help="用浏览器语音朗读（Chrome / Edge 支持中文）",
    ):
        st.session_state["_voice_queue"] = texts
    played_flag = f"_voice_played_{key}"
    if auto and not st.session_state.get(played_flag):
        st.session_state["_voice_queue"] = texts
        st.session_state[played_flag] = True
    queue = st.session_state.get("_voice_queue") or []
    if queue:
        _emit_speech(queue)
        st.session_state["_voice_queue"] = []

# 报告跟随当前题目与输入文件；切换题目/文件后自动清空旧报告
report_key = (
    question["id"],
    category,
    audio_file.name if audio_file else None,
    video_file.name if video_file else None,
)
if st.session_state.get("report_key") != report_key:
    st.session_state["report"] = None
    st.session_state["report_key"] = report_key

def _run_single_evaluation() -> None:
    """单题评估：保存上传文件 → 门面分析 → 写入报告状态 → 清理临时文件。"""
    st.session_state["report"] = None
    audio_path = _save_temp(audio_file, ".wav") if audio_file else None
    video_path = _save_temp(video_file, ".mp4") if video_file else None
    try:
        facade = InterviewCoachFacade()
        with st.spinner(
            "正在提取音频并分析，请稍候…" if audio_file is None
            else "多模态分析进行中，请稍候…"
        ):
            st.session_state["report"] = facade.run_interview(
                question=question, audio_path=audio_path,
                video_path=video_path, job_category=category,
                username=auth_user,
                candidate_context={
                    key: knowledge.get(key, "")
                    for key in ("candidate_profile", "project_evidence", "job_description")
                    if knowledge.get(key)
                })
    except InputValidationError as exc:
        st.error(f"输入不可用：{exc}")
    except AnalysisBusyError as exc:
        st.warning(str(exc))
    except Exception as exc:  # 兜底：避免把完整堆栈直接暴露在页面上
        st.error(f"分析失败：{exc}")
    finally:
        # 用户上传的临时文件在分析结束后删除；提取出的音频由门面清理
        for path in (audio_path, video_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


can_evaluate = audio_file is not None or video_file is not None
button_cols = st.columns([1, 1, 2])
with button_cols[0]:
    if st.button("开始评估", type="primary", disabled=not can_evaluate):
        _run_single_evaluation()
with button_cols[1]:
    _voice(f"single_q_{question['id']}", [question.get("content", "")], auto=False)

report = st.session_state.get("report")

# ---------------------------------------------------------------------------
# 报告展示
# ---------------------------------------------------------------------------
if report is not None:
    st.markdown('<div class="section-head">评估报告</div>', unsafe_allow_html=True)
    report_input_meta = getattr(report, "input_meta", {})
    report_confidence = float(getattr(report, "confidence", 1.0))
    if report_input_meta.get("audio_source") == "video":
        st.caption("本次语音与内容分析使用从视频自动提取的音轨。")
    if report_confidence < 0.6:
        st.caption("本次多个模态降级，综合分与五维指标仅供参考。")

    # 综合分 + 三维度
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.markdown(
            f'<div class="prism-card" style="text-align:center">'
            f'<div class="score-label">综合评分</div>'
            f'<div class="score-big">{report.overall_score}</div>'
            f'<div class="card-sub" style="margin-top:0.5rem">满分 100</div>'
            f'<div class="card-sub">评分置信度 {int(round(report_confidence * 100))}%</div>'
            f'</div>', unsafe_allow_html=True)
    with c2:
        labels = {"content": "内容质量", "speech": "语音表达", "vision": "画面可见度（不计总分）"}
        rows = ""
        for mod, name in labels.items():
            score = report.dimension_scores.get(mod)
            status = report.modality_status.get(mod, "disabled")
            badge = ("badge-ok", "正常") if status == "ok" else ("badge-deg", "降级")
            rows += (f'<div class="dim-row"><span class="dim-name">{name}'
                     f'<span class="badge {badge[0]}">{badge[1]}</span></span>'
                     f'<span class="dim-val">{score if score is not None else "—"}</span></div>')
        st.markdown(f'<div class="prism-card">{rows}</div>', unsafe_allow_html=True)

    # 雷达图
    try:
        import plotly.graph_objects as go
        dims = list(report.radar.keys())
        vals = list(report.radar.values())
        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]],
            fill="toself", name="五维评估",
            line=dict(color="#5a9b7e", width=2),
            fillcolor="rgba(90,155,126,0.25)"))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#e0e8e0"),
                       bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
            height=360, margin=dict(t=20, b=20, l=40, r=40),
            showlegend=False)
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.markdown('<div class="card-title">五维能力雷达</div>', unsafe_allow_html=True)
            for k, v in report.radar.items():
                bar_w = int(v)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin:0.4rem 0;font-size:0.9rem">'
                    f'<span style="color:var(--text)">{k}</span>'
                    f'<span style="font-weight:700;color:var(--green-2)">{v}</span></div>'
                    f'<div style="background:var(--green-bg);border-radius:6px;height:8px">'
                    f'<div style="width:{bar_w}%;height:8px;background:var(--green);'
                    f'border-radius:6px"></div></div>',
                    unsafe_allow_html=True)
    except ImportError:
        st.bar_chart(report.radar)

    # 改进建议
    st.markdown('<div class="section-head">改进建议</div>', unsafe_allow_html=True)
    for s in report.suggestions:
        st.markdown(f'<div class="sug-item">{s}</div>', unsafe_allow_html=True)

    scoring_details = getattr(report, "scoring_details", {})
    if scoring_details.get("evidence") or getattr(report, "transcript", ""):
        with st.expander("查看转写、评分证据与扣分依据"):
            if report.transcript:
                st.text_area(
                    "语音转写（本次页面展示；历史记录不保存转写）",
                    value=report.transcript,
                    height=140,
                    disabled=True,
                    key=f"transcript_{report.session_id}",
                )
            st.caption(
                "评估方式：" +
                ("本地 AI" if scoring_details.get("method") == "ollama" else "规则降级")
            )
            checklist = scoring_details.get("checklist") or {}
            if checklist:
                st.markdown(
                    "**量化指标命中清单**"
                    "（每项 20 分；✅ 命中 · ◐ 部分命中 · ⬜ 未命中）"
                )
                rubric_names = scoring_details.get("rubric") or {}
                for dim_key, payload in checklist.items():
                    dim_name = (rubric_names.get(dim_key) or {}).get("name", dim_key)
                    badges = []
                    for item in payload.get("hits", []):
                        badges.append(
                            f'<span style="background:#e8f3ee;color:#4a8c6e;'
                            f'border-radius:8px;padding:1px 8px;margin-right:4px;'
                            f'font-size:0.75rem">✅ {item["id"]} {item["name"]}</span>'
                        )
                    for item in payload.get("partial", []):
                        badges.append(
                            f'<span style="background:#fdf0e6;color:#c97a2b;'
                            f'border-radius:8px;padding:1px 8px;margin-right:4px;'
                            f'font-size:0.75rem">◐ {item["id"]} {item["name"]}</span>'
                        )
                    for item in payload.get("missed", []):
                        badges.append(
                            f'<span style="background:#f3f5f3;color:#8aa098;'
                            f'border-radius:8px;padding:1px 8px;margin-right:4px;'
                            f'font-size:0.75rem">⬜ {item["id"]} {item["name"]}</span>'
                        )
                    st.markdown(
                        f'<div style="margin:0.35rem 0">'
                        f'<b>{dim_name}：{payload.get("score", 0)} 分</b><br>'
                        f'{"".join(badges)}</div>',
                        unsafe_allow_html=True,
                    )
            st.json({
                "逐维证据": scoring_details.get("evidence", {}),
                "证据缺口": scoring_details.get("gaps", {}),
            })

    with st.expander("查看各模态耗时、状态与输入信息"):
        st.json({
            "status": report.modality_status,
            "elapsed_sec": report.elapsed,
            "errors": getattr(report, "errors", {}),
            "input": report_input_meta,
        })

    if st.button("下一题", key="next_question", use_container_width=True):
        q_list = by_category(category)
        q_ids = [q["id"] for q in q_list]
        idx = q_ids.index(question["id"]) if question["id"] in q_ids else 0
        st.session_state.selected_qid = q_ids[(idx + 1) % len(q_ids)]
        st.session_state.pop("report", None)
        st.session_state.pop("report_key", None)
        st.rerun()

# ---------------------------------------------------------------------------
# 📞 微信通话式 AI 语音面试
# ---------------------------------------------------------------------------

def _call_context(knowledge):
    return {
        key: knowledge.get(key, "")
        for key in ("candidate_profile", "project_evidence", "job_description")
        if knowledge.get(key)
    }


def _call_question_msg(q, round_no, limit, feedback="", hint=""):
    """下发「AI 提问」消息：iframe 展示题目/反馈并进入播报动画。"""
    return {
        "type": "turn",
        "phase": "speak",
        "round": round_no,
        "total": limit,
        "question": q.get("content", ""),
        "feedback": feedback,
        "hint": hint,
    }


def _call_listen_msg(q, round_no, limit):
    """TTS 播报结束后下发的「请回答」消息：iframe 打开麦克风开始录音。"""
    return {
        "type": "turn",
        "phase": "listen",
        "round": round_no,
        "total": limit,
        "question": q.get("content", ""),
    }


def _call_save_review(auth_user, turns, knowledge_prefix):
    """通话结束：把整场复盘沉淀追加进知识库（与多轮面试区逻辑一致）。"""
    try:
        summary = review_summary(turns)
        kb = load_knowledge(str(auth_user))
        previous = str(kb.get("interview_review", "")).strip()
        merged = f"{summary}\n\n{previous}" if previous else summary
        kb["interview_review"] = merged[:8000]
        save_knowledge(str(auth_user), kb)
        st.session_state[f"{knowledge_prefix}_review_draft"] = merged[:8000]
        st.session_state[f"{knowledge_prefix}_review_ver"] = (
            st.session_state.get(f"{knowledge_prefix}_review_ver", 0) + 1
        )
    except (OSError, ValueError):
        pass


def _defer_broadcast(display, speak=None, listen=None):
    """把下行广播暂存到 session_state，交由下一次 rerun 开头渲染。

    事件处理紧跟着 st.rerun()：同一次 run 内调用的 st.html 增量会被
    Streamlit 丢弃（实测 DOM 中不出现），因此广播必须延迟到下一个
    完整 run（该 run 正常结束、增量正常刷新）再注入。
    """
    st.session_state["call_pending_broadcast"] = {
        "display": display,
        "speak": speak or [],
        "listen": listen,
    }


def _handle_call_event(ev, auth_user, knowledge, knowledge_prefix, first_question):
    """处理通话 iframe 回传事件，推进对话状态并广播下一轮消息。"""
    kind = ev.get("event") or ""
    q = st.session_state.get("call_question") or {}
    turns = list(st.session_state.get("call_turns", []))
    limit = int(st.session_state.get("call_limit") or 4)
    round_no = len(turns) + 1
    session_id = st.session_state.get("call_session_id", "call")
    q_text = q.get("content", "")
    st.session_state["call_notice"] = None

    if kind == "ready":
        # 首次接听 → 开场白 + 第一题；中途重连（如麦克风恢复）→ 重播当前题
        st.session_state["call_started"] = True
        speak = []
        if not st.session_state.get("call_opening_done"):
            speak.append(opening_text(limit, q_text))
            st.session_state["call_opening_done"] = True
        speak.append(q_text)
        _defer_broadcast(
            _call_question_msg(q, round_no, limit),
            speak=speak,
            listen=_call_listen_msg(q, round_no, limit),
        )
        return

    if kind == "audio":
        try:
            wav_bytes = decode_wav_b64(str(ev.get("data") or ""))
        except (ValueError, TypeError):
            st.session_state["call_notice"] = "⚠️ 音频数据接收失败，已请求重新作答。"
            _defer_broadcast(
                _call_question_msg(q, round_no, limit, hint="音频接收失败，请再回答一次。"),
                speak=["音频接收失败，请再回答一次。", q_text],
                listen=_call_listen_msg(q, round_no, limit),
            )
            return
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)
        report = None
        context = _call_context(knowledge)
        try:
            with st.spinner("正在转写并分析你的回答…"):
                report = InterviewCoachFacade().run_interview(
                    question=q,
                    audio_path=audio_path,
                    video_path=None,
                    job_category=q.get("category", "default"),
                    username=auth_user,
                    candidate_context=context,
                )
        except (InputValidationError, AnalysisBusyError) as exc:
            st.session_state["call_notice"] = f"⚠️ {exc}"
        except Exception as exc:
            st.session_state["call_notice"] = f"本轮处理失败：{exc}"
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass
        if report is None:
            # 分析失败：原题重答，不消耗轮次
            _defer_broadcast(
                _call_question_msg(q, round_no, limit, hint="本轮分析遇到问题，请再回答一次。"),
                speak=["本轮分析遇到问题，请再回答一次。", q_text],
                listen=_call_listen_msg(q, round_no, limit),
            )
            return
        turn = {
            "question": q_text,
            "transcript": report.transcript,
            "report": report,
        }
        turns.append(turn)
        st.session_state["call_turns"] = turns
        if st.session_state.get("call_persist"):
            save_interview_turn(
                session_id, round_no, auth_user, q_text, turn["transcript"], report
            )
        if round_no >= limit:
            closing = closing_text(turns)
            if st.session_state.get("call_persist"):
                _call_save_review(auth_user, turns, knowledge_prefix)
            avg = round(
                sum(float(t["report"].overall_score) for t in turns) / len(turns), 1
            )
            _defer_broadcast(
                {"type": "end", "closing": closing, "avg": avg, "total": len(turns)},
                speak=[closing],
                listen=None,
            )
            return
        follow_up_context = dict(context)
        if knowledge.get("interviewer_style"):
            follow_up_context["interviewer_style"] = knowledge["interviewer_style"]
        follow_up = generate_follow_up(
            q,
            report.transcript,
            candidate_context=follow_up_context,
            prior_turns=turns,
            report=report,
        )
        new_q = {
            "id": f"C{round_no + 1}",
            "category": q.get("category", "综合"),
            "content": follow_up["question"],
        }
        st.session_state["call_question"] = new_q
        _defer_broadcast(
            _call_question_msg(
                new_q, round_no + 1, limit, feedback=feedback_voice_text(report)
            ),
            speak=[feedback_voice_text(report), new_q["content"]],
            listen=_call_listen_msg(new_q, round_no + 1, limit),
        )
        return

    if kind == "no_speech":
        # 静音/回答过短：重播同题，不消耗轮次
        _defer_broadcast(
            _call_question_msg(q, round_no, limit, hint="我没有听清你的回答，请再回答一次。"),
            speak=["我没有听清你的回答，请再回答一次。", q_text],
            listen=_call_listen_msg(q, round_no, limit),
        )
        return

    if kind == "hangup":
        if turns:
            closing = closing_text(turns)
            if st.session_state.get("call_persist"):
                _call_save_review(auth_user, turns, knowledge_prefix)
            avg = round(
                sum(float(t["report"].overall_score) for t in turns) / len(turns), 1
            )
            _defer_broadcast(
                {"type": "end", "closing": closing, "avg": avg, "total": len(turns)},
                speak=[closing],
                listen=None,
            )
        else:
            _defer_broadcast(
                {"type": "end", "closing": "本次通话未开始正式问答。", "avg": None, "total": 0},
                speak=None,
                listen=None,
            )
        return

    if kind == "restart":
        # 再来一场：重置会话，iframe 回到响铃界面等待再次接听
        st.session_state["call_session_id"] = uuid.uuid4().hex[:12]
        st.session_state["call_turns"] = []
        st.session_state["call_question"] = dict(first_question)
        st.session_state["call_started"] = False
        st.session_state["call_opening_done"] = False
        st.session_state["call_notice"] = None
        _defer_broadcast({"type": "reset"}, speak=None, listen=None)
        return

    if kind == "quit":
        st.session_state["call_active"] = False
        st.session_state["call_started"] = False
        st.session_state["call_notice"] = None
        return

    if kind == "mic_error":
        st.session_state["call_notice"] = (
            "麦克风不可用：" + str(ev.get("message") or "请检查浏览器权限后重试。")
        )


st.markdown('<div class="section-head">📞 微信通话式 AI 语音面试</div>', unsafe_allow_html=True)
st.caption(
    "像微信通话一样模拟真人面试：点击「开始通话」后接听来电，AI 面试官语音提问，"
    "你直接开口回答，说完自动停止录音并生成评分反馈与追问，全程免手动操作。"
    "原始音视频仅用于本次分析，不写入数据库。"
)

call_limit = st.slider("通话轮数", 2, 6, 4, key="call_round_limit")
call_persist = st.checkbox(
    "将通话问答转写和评分保存到本机知识数据库（不保存音视频）",
    value=True,
    key=f"call_persist_{auth_user}",
)

if st.session_state.get("call_user") != auth_user:
    for key in _CALL_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["call_user"] = auth_user

call_turns = st.session_state.get("call_turns", [])
call_active = bool(st.session_state.get("call_active"))

if not call_active:
    start_col, _ = st.columns([1, 3])
    with start_col:
        if st.button("📞 开始通话", type="primary", key="call_start_btn"):
            st.session_state["call_session_id"] = uuid.uuid4().hex[:12]
            st.session_state["call_turns"] = []
            st.session_state["call_question"] = dict(question)
            st.session_state["call_limit"] = call_limit
            st.session_state["call_persist"] = call_persist
            st.session_state["call_active"] = True
            st.session_state["call_started"] = False
            st.session_state["call_opening_done"] = False
            st.session_state["call_notice"] = None
            st.session_state["call_last_sig"] = None
            st.rerun()
else:
    # 上一轮事件处理暂存的广播：在本 run 开头渲染（本 run 正常结束，
    # 增量不会被 st.rerun() 丢弃），再继续处理组件新回传
    pending_b = st.session_state.pop("call_pending_broadcast", None)
    if pending_b:
        broadcast(
            pending_b["display"],
            speak=pending_b["speak"],
            listen=pending_b["listen"],
        )
    st.caption(
        f"通话中：{st.session_state.get('call_session_id', '')}｜"
        f"已完成 {len(call_turns)}/{st.session_state.get('call_limit', call_limit)} 轮"
    )
    ev = parse_event(render_call())
    if ev:
        # 组件回传值在 rerun 间持久保留，用事件序号去重：
        # 每个事件只处理一次，重复 rerun 不会再次广播/推进状态
        sig = f"{ev.get('event')}:{ev.get('seq')}"
        if sig != st.session_state.get("call_last_sig"):
            st.session_state["call_last_sig"] = sig
            _handle_call_event(ev, auth_user, knowledge, knowledge_prefix, question)
            st.rerun()
    if st.session_state.get("call_notice"):
        st.warning(st.session_state["call_notice"])
    if st.button("退出通话模式", key="call_exit_btn"):
        st.session_state["call_active"] = False
        st.session_state["call_started"] = False
        st.session_state["call_notice"] = None
        st.rerun()

if call_turns:
    if not call_active:
        saved_note = (
            "本场问答与评分已保存到本机知识库（可在「多轮语音/视频面试」区的"
            "「查看已保存的多轮面试」中打开）。"
            if st.session_state.get("call_persist")
            else "本场未开启保存，记录仅保留在当前页面。"
        )
        st.caption(f"本次通话共 {len(call_turns)} 轮。{saved_note}")
    for turn_index, turn in enumerate(call_turns, start=1):
        with st.chat_message("assistant"):
            st.markdown(f"**第 {turn_index} 题**")
            st.write(turn["question"])
        with st.chat_message("user"):
            st.write(turn.get("transcript") or "未能获得语音转写。")
            st.metric("本轮能力评分", f"{turn['report'].overall_score}/100")
            with st.expander(f"第 {turn_index} 轮反馈与证据"):
                for suggestion in turn["report"].suggestions:
                    st.markdown(f"- {suggestion}")
                st.json(turn["report"].scoring_details)

# ---------------------------------------------------------------------------
# 多轮语音/视频模拟面试
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head">多轮语音 / 视频面试</div>', unsafe_allow_html=True)
st.caption(
    "面试官问题可朗读；用浏览器麦克风录制回答，并可附加视频。"
    "每轮回答后会生成针对性追问。原始音视频仅用于本次分析，不写入数据库。"
)

round_limit = st.slider("本场最多轮数", 2, 10, 5, key="dialog_round_limit")
persist_dialog = st.checkbox(
    "将本场问答转写和评分保存到本机知识数据库（不保存音视频）",
    value=True,
    key=f"dialog_persist_{auth_user}",
)
voice_enabled = st.toggle(
    "AI 面试官语音播报",
    value=True,
    key="dialog_voice",
    help="AI 面试官会朗读开场白、题目、本轮反馈与结束语；"
         "浏览器拦截自动播放时可点🔊按钮。",
)
if not voice_enabled:
    st.session_state["_voice_queue"] = []

if st.session_state.get("dialog_user") != auth_user:
    for key in ("dialog_user", "dialog_session_id", "dialog_turns",
                "dialog_question", "dialog_active", "dialog_follow_up_source",
                "dialog_opening", "dialog_opening_done"):
        st.session_state.pop(key, None)
    st.session_state["dialog_user"] = auth_user

if st.button("开始新的多轮面试", type="primary", key="dialog_start"):
    st.session_state["dialog_session_id"] = uuid.uuid4().hex[:12]
    st.session_state["dialog_turns"] = []
    st.session_state["dialog_question"] = dict(question)
    st.session_state["dialog_active"] = True
    st.session_state["dialog_persist"] = persist_dialog
    st.session_state["dialog_follow_up_source"] = ""
    st.session_state["dialog_opening"] = opening_text(
        round_limit, question.get("content", "")
    )
    st.session_state["dialog_opening_done"] = False

dialog_turns = st.session_state.get("dialog_turns", [])
if st.session_state.get("dialog_session_id"):
    st.caption(
        f"当前会话：{st.session_state['dialog_session_id']}｜"
        f"已完成 {len(dialog_turns)}/{round_limit} 轮"
        + ("｜问答文本已保存" if st.session_state.get("dialog_persist") else "｜仅保存在当前页面")
    )
    for turn_index, turn in enumerate(dialog_turns, start=1):
        with st.chat_message("assistant"):
            st.markdown(f"**第 {turn_index} 题**")
            st.write(turn["question"])
        with st.chat_message("user"):
            st.write(turn.get("transcript") or "未能获得语音转写。")
            turn_report = turn["report"]
            st.metric("本轮能力评分", f"{turn_report.overall_score}/100")
            st.caption(
                f"内容 {turn_report.dimension_scores.get('content', '—')} 分 · "
                f"语音 {turn_report.dimension_scores.get('speech', '—')} 分 · "
                f"画面可见度 {turn_report.dimension_scores.get('vision', '—')} 分（不计总分）"
            )
            if voice_enabled:
                _voice(
                    f"fb_{st.session_state['dialog_session_id']}_{turn_index}",
                    [feedback_voice_text(turn_report)],
                    auto=False,
                )
            with st.expander(f"第 {turn_index} 轮反馈与证据"):
                st.write("**改进建议**")
                for suggestion in turn_report.suggestions:
                    st.markdown(f"- {suggestion}")
                st.json(turn_report.scoring_details)

    if st.session_state.get("dialog_active"):
        current_question = st.session_state["dialog_question"]
        current_round = len(dialog_turns) + 1
        st.markdown(f"**第 {current_round} 轮 · {current_question.get('category', '综合')}**")
        st.info(current_question.get("content", ""))
        if st.session_state.get("dialog_follow_up_source"):
            st.caption(f"追问来源：{st.session_state['dialog_follow_up_source']}")
        if voice_enabled:
            if st.session_state.get("dialog_opening") and not st.session_state.get(
                "dialog_opening_done"
            ):
                st.session_state["dialog_opening_done"] = True
                voice_texts = [
                    st.session_state["dialog_opening"],
                    current_question.get("content", ""),
                ]
            else:
                voice_texts = [current_question.get("content", "")]
            _voice(
                f"q_{st.session_state['dialog_session_id']}_{current_round}",
                voice_texts,
                auto=True,
            )
        audio_reply = st.audio_input(
            "录制本轮语音回答",
            key=f"dialog_audio_{st.session_state['dialog_session_id']}_{current_round}",
        )
        video_reply = st.file_uploader(
            "附加本轮视频（可选）",
            type=["mp4", "mov"],
            key=f"dialog_video_{st.session_state['dialog_session_id']}_{current_round}",
            help="视频会参与画面可见度分析；不会用于推断情绪、人格或面试能力。",
        )
        if st.button(
            "提交回答并生成追问",
            key=f"dialog_submit_{st.session_state['dialog_session_id']}_{current_round}",
            disabled=audio_reply is None and video_reply is None,
        ):
            audio_path = _save_temp(audio_reply, ".wav") if audio_reply else None
            video_path = _save_temp(video_reply, ".mp4") if video_reply else None
            try:
                context = {
                    key: knowledge.get(key, "")
                    for key in ("candidate_profile", "project_evidence", "job_description")
                    if knowledge.get(key)
                }
                with st.spinner("正在转写回答并生成多模态反馈…"):
                    turn_report = InterviewCoachFacade().run_interview(
                        question=current_question,
                        audio_path=audio_path,
                        video_path=video_path,
                        job_category=current_question.get("category", "default"),
                        username=auth_user,
                        candidate_context=context,
                    )
                turn = {
                    "question": current_question.get("content", ""),
                    "transcript": turn_report.transcript,
                    "report": turn_report,
                }
                st.session_state["dialog_turns"].append(turn)
                if st.session_state.get("dialog_persist"):
                    save_interview_turn(
                        st.session_state["dialog_session_id"],
                        current_round,
                        auth_user,
                        turn["question"],
                        turn["transcript"],
                        turn_report,
                    )
                if current_round >= round_limit:
                    st.session_state["dialog_active"] = False
                    if st.session_state.get("dialog_persist"):
                        # 知识库沉淀：整场复盘要点自动追加到用户知识库
                        try:
                            summary = review_summary(st.session_state["dialog_turns"])
                            kb = load_knowledge(str(auth_user))
                            previous = str(kb.get("interview_review", "")).strip()
                            merged = f"{summary}\n\n{previous}" if previous else summary
                            kb["interview_review"] = merged[:8000]
                            save_knowledge(str(auth_user), kb)
                            st.session_state[f"{knowledge_prefix}_review_draft"] = merged[:8000]
                            st.session_state[f"{knowledge_prefix}_review_ver"] = (
                                st.session_state.get(f"{knowledge_prefix}_review_ver", 0) + 1
                            )
                        except (OSError, ValueError):
                            pass
                else:
                    follow_up_context = dict(context)
                    if knowledge.get("interviewer_style"):
                        follow_up_context["interviewer_style"] = knowledge["interviewer_style"]
                    follow_up = generate_follow_up(
                        current_question,
                        turn_report.transcript,
                        candidate_context=follow_up_context,
                        prior_turns=st.session_state["dialog_turns"],
                        report=turn_report,
                    )
                    st.session_state["dialog_question"] = {
                        "id": f"D{current_round + 1}",
                        "category": current_question.get("category", "综合"),
                        "content": follow_up["question"],
                    }
                    st.session_state["dialog_follow_up_source"] = follow_up["source"]
                st.rerun()
            except InputValidationError as exc:
                st.error(f"输入不可用：{exc}")
            except AnalysisBusyError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"本轮处理失败：{exc}")
            finally:
                for path in (audio_path, video_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
    elif dialog_turns:
        st.success("本场面试已完成。你可以查看上方逐轮反馈，或开始新的一场。")
        if voice_enabled:
            _voice(
                f"close_{st.session_state['dialog_session_id']}",
                [closing_text(dialog_turns)],
                auto=True,
            )

saved_dialogs = list_interview_sessions(auth_user)
if saved_dialogs:
    with st.expander("查看已保存的多轮面试"):
        dialog_options = {
            item["session_id"]: (
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(item['created_at']))}"
                f" · {item['turn_count']} 轮 · 均分 {item['average_score']}"
            )
            for item in saved_dialogs
        }
        selected_dialog = st.selectbox(
            "选择历史对话",
            list(dialog_options),
            format_func=lambda sid: dialog_options[sid],
            key="saved_dialog_select",
        )
        for saved_turn in load_interview_turns(selected_dialog, auth_user):
            st.markdown(f"**題目：** {saved_turn['question']}")
            st.write(saved_turn["transcript"] or "无可用转写")
            st.caption(f"本轮评分：{saved_turn['overall']}/100")

# ---------------------------------------------------------------------------
# 历史趋势
# ---------------------------------------------------------------------------
history = load_history(username=auth_user)
if history:
    st.markdown('<div class="section-head">历史练习趋势</div>', unsafe_allow_html=True)
    st.line_chart({"综合评分": [h["overall"] for h in history]})

    with st.expander("查看历史报告详情"):
        options = {
            h["session_id"]: (
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(h['created_at']))}"
                f" · {h['overall']} 分"
            )
            for h in reversed(history)
        }
        selected = st.selectbox(
            "选择一次练习",
            list(options.keys()),
            format_func=lambda sid: options[sid],
            key="history_select",
        )
        if st.button("加载所选报告", key="history_load"):
            payload = load_report(selected, username=auth_user)
            st.session_state["history_report"] = payload
            st.session_state["history_report_sid"] = selected
            st.session_state["history_report_missing"] = payload is None

        if st.session_state.get("history_report_sid") == selected:
            payload = st.session_state.get("history_report")
            if payload:
                question_text = payload.get("question", {}).get("content", "")
                if question_text:
                    st.markdown(f"**题目：** {question_text}")
                c1, c2 = st.columns(2)
                c1.metric("综合评分", payload.get("overall_score", "—"))
                confidence = float(payload.get("confidence", 0.0) or 0.0)
                c2.metric("评分置信度", f"{int(round(confidence * 100))}%")
                labels = {
                    "content": "内容质量",
                    "speech": "表达能力",
                    "vision": "精神状态",
                }
                for key, score in payload.get("dimension_scores", {}).items():
                    st.write(f"{labels.get(key, key)}：{score}")
                for suggestion in payload.get("suggestions", [])[:8]:
                    st.markdown(f"- {suggestion}")
                with st.expander("完整报告 JSON"):
                    st.json(payload)
            elif st.session_state.get("history_report_missing"):
                st.warning("该记录生成于完整报告持久化之前，暂无报告 JSON。")
        elif st.session_state.get("history_report_sid"):
            st.caption("点击“加载所选报告”查看详情。")
