"""登录/注册页：沿用主应用的浅鼠尾草绿主题。"""
from __future__ import annotations

import time

import streamlit as st

from data.auth import AuthError, authenticate, create_user, has_users
from data.storage import claim_legacy_results

LOGIN_CSS = """
<style>
/* 登录页隐藏侧边栏，内容居中 */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stMainBlockContainer"] {
  max-width: 980px !important;
  padding-top: 2.5rem !important;
}
[data-testid="stTabs"] { max-width: 460px; margin: 0 auto; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 1.4rem;
  border-bottom: 1px solid #e5e7eb;
  margin-bottom: 1.1rem;
}
[data-testid="stTabs"] button[role="tab"] {
  color: #8a8f98 !important;
  font-weight: 600;
  font-size: 0.95rem;
  padding: 0.35rem 0.1rem 0.55rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--red, #d6606b) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background-color: var(--red, #d6606b) !important;
  height: 2px;
}
[data-testid="stForm"] {
  max-width: 460px;
  margin: 0 auto;
  background: #ffffff;
  border: 1px solid var(--border, #e0e8e0);
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(45, 52, 54, 0.06);
  padding: 1.35rem 1.5rem 1.25rem;
}
[data-testid="stForm"] label {
  color: var(--text, #2d3436) !important;
  font-weight: 600;
  font-size: 0.9rem;
}
[data-testid="stForm"] input {
  background: #fbfcfb !important;
  border-radius: 8px !important;
}
[data-testid="stForm"] input:focus {
  border-color: var(--red, #d6606b) !important;
  box-shadow: 0 0 0 1px var(--red, #d6606b) !important;
}
[data-testid="stForm"] button {
  width: 100%;
  background: #ffffff !important;
  color: var(--text, #2d3436) !important;
  border: 1px solid var(--border, #e0e8e0) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1rem !important;
}
[data-testid="stForm"] button:hover {
  background: var(--green-bg, #e8f3ee) !important;
  border-color: var(--green, #5a9b7e) !important;
  color: var(--green-2, #4a8c6e) !important;
}
</style>
"""


def _clear_report_state() -> None:
    for key in (
        "report", "report_key",
        "history_report", "history_report_sid", "history_report_missing",
    ):
        st.session_state.pop(key, None)


def _finish_login(username: str, claim_legacy: bool = True) -> None:
    """统一处理登录成功后的会话状态与旧数据认领。"""
    st.session_state["auth_user"] = username
    st.session_state["auth_last_active"] = time.time()
    if claim_legacy:
        try:
            claim_legacy_results(username)
        except Exception:
            # 旧数据认领失败不应阻塞登录
            pass
    _clear_report_state()
    st.rerun()


def render_login() -> None:
    """渲染登录/注册页并在完成后停止脚本，避免进入主应用。"""
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1.5, 1])

    with middle:
        login_tab, register_tab = st.tabs(["登录", "注册"])

        with login_tab:
            try:
                first_use = not has_users()
            except Exception:
                first_use = False
            if first_use:
                st.caption("首次使用，请先切换到“注册”创建账号。")

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input(
                    "用户名", key="login_user", placeholder="请输入用户名"
                )
                password = st.text_input(
                    "密码", type="password", key="login_pwd",
                    placeholder="请输入密码",
                )
                submitted = st.form_submit_button("登录", use_container_width=True)

            if submitted:
                name = username.strip()
                if not name or not password:
                    st.error("请输入用户名和密码")
                else:
                    try:
                        result = authenticate(name, password)
                    except Exception as exc:
                        st.error(f"登录失败：{exc}")
                    else:
                        if result.ok:
                            _finish_login(result.username or name)
                        else:
                            st.error(result.message)

        with register_tab:
            try:
                first_user = not has_users()
            except Exception:
                first_user = False
            with st.form("register_form", clear_on_submit=False):
                reg_user = st.text_input(
                    "用户名", key="reg_user", placeholder="2-32 个字符"
                )
                reg_pwd = st.text_input(
                    "密码", type="password", key="reg_pwd",
                    placeholder="至少 8 位",
                )
                reg_pwd2 = st.text_input(
                    "确认密码", type="password", key="reg_pwd2",
                    placeholder="再次输入密码",
                )
                reg_submit = st.form_submit_button("注册", use_container_width=True)

            if reg_submit:
                if reg_pwd != reg_pwd2:
                    st.error("两次输入的密码不一致")
                else:
                    try:
                        name = create_user(reg_user, reg_pwd)
                    except AuthError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"注册失败：{exc}")
                    else:
                        # 只有首个账号认领旧数据；后续账号不会抢走原有历史
                        _finish_login(name, claim_legacy=first_user)

    st.stop()
