# config.py
import streamlit as st
import matplotlib.pyplot as plt


# 页面配置
def setup_page_config():
    st.set_page_config(
        page_title="医学科研AutoML平台",
        layout="wide",
        page_icon="🏥"
    )

    # 设置中文字体，解决图表中文显示问题
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.rcParams["figure.dpi"] = 120  # 优化图表清晰度
    plt.rcParams["figure.figsize"] = (10, 6)  # 默认图表大小


# 自定义样式常量
CARD_STYLE = """
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
"""

# 定义固定操作流程（用于步骤导航）
PAGE_FLOW = [
    "数据导入与预览",
    "变量定义与任务设置",
    "模型训练与配置",
    "模型评估与解释",
    "模型预测",
    "模型管理"
]


# 会话状态初始化
def init_session_state():
    # 确保所有需要持久化的状态都被初始化
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'features' not in st.session_state:
        st.session_state.features = []
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'actual_model' not in st.session_state:
        st.session_state.actual_model = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'task_type' not in st.session_state:
        st.session_state.task_type = "分类"
    if 'saved_models' not in st.session_state:
        st.session_state.saved_models = []
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model_training_mode' not in st.session_state:
        st.session_state.model_training_mode = "选择单个模型"
    if 'all_features' not in st.session_state:
        st.session_state.all_features = []
    # SHAP相关状态
    if 'shap_success' not in st.session_state:
        st.session_state.shap_success = False
    # 新增：当前页面状态（默认从第一个页面开始）
    if 'current_page' not in st.session_state:
        st.session_state.current_page = PAGE_FLOW[0]
