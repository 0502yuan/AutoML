# main.py
import streamlit as st
from config import setup_page_config, init_session_state, CARD_STYLE, PAGE_FLOW
# 修复导入路径（确保与文件结构一致）
from pages_functions.data_source import data_source_page
from pages_functions.variable_definition import variable_definition_page
from pages_functions.model_training import model_training_page
from pages_functions.model_evaluation import model_evaluation_page
from pages_functions.model_prediction import model_prediction_page
from pages_functions.model_management import save_model_page


def main():
    # 初始化配置
    setup_page_config()
    init_session_state()

    # 将样式常量和页面流程存入会话状态
    st.session_state.card_style = CARD_STYLE
    st.session_state.page_flow = PAGE_FLOW

    # 侧边栏导航（与当前页面状态同步）
    with st.sidebar:
        # 修复图片路径（建议使用相对路径）
        try:
            st.image("logo/logo.png", use_container_width=True)  # 假设logo在项目根目录的logo文件夹下
        except:
            st.image("https://via.placeholder.com/300x100?text=医学AI平台", use_container_width=True)

        # 核心修复：用HTML标签固定标题宽度，避免换行
        st.markdown(
            """
            <div style="width: 280px; white-space: nowrap;">
                <h1 style="margin: 0; padding: 0;">医学科研AutoML平台</h1>
            </div>
            """,
            unsafe_allow_html=True  # 允许解析HTML标签
        )

        st.write("医学数据建模与分析工具")

        st.markdown("---")
        st.subheader("导航菜单")
        # 侧边栏选中状态与current_page同步
        selected_page = st.radio(
            "",
            PAGE_FLOW,
            index=PAGE_FLOW.index(st.session_state.current_page) if st.session_state.current_page in PAGE_FLOW else 0
        )

        # 侧边栏切换页面
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()

        st.markdown("---")
        st.info("""
        **版本**: 1.2.0  
        **© 2025 医学AI研究团队**  
        专为医学科研设计的自动化机器学习平台
        """)

    # 页面路由（修复页面函数映射）
    current_page = st.session_state.current_page
    if current_page == "数据导入与预览":
        data_source_page()
    elif current_page == "变量定义与任务设置":
        variable_definition_page()
    elif current_page == "模型训练与配置":
        model_training_page()
    elif current_page == "模型评估与解释":
        model_evaluation_page()
    elif current_page == "模型预测":
        model_prediction_page()
    elif current_page == "模型管理":
        save_model_page()


if __name__ == "__main__":
    main()