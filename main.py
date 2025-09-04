# main.py
import streamlit as st
from config import setup_page_config, init_session_state, CARD_STYLE, PAGE_FLOW
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
        st.image("D:\Projects\AutoML_1\logo\logo.png", use_container_width=True)
        st.title("医学科研AutoML平台")
        st.write("医学数据建模与分析工具")

        st.markdown("---")
        st.subheader("导航菜单")
        # 侧边栏选中状态与current_page同步
        selected_page = st.radio(
            "",
            PAGE_FLOW,
            index=PAGE_FLOW.index(st.session_state.current_page)  # 同步选中状态
        )

        # 侧边栏切换页面时更新current_page（无需rerun，Streamlit会自动重渲染）
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()

        st.markdown("---")
        st.info("""
        **版本**: 1.1.0  
        **© 2025 医学AI研究团队**  
        专为医学科研设计的自动化机器学习平台
        """)

    # 页面路由：根据current_page加载对应页面
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
