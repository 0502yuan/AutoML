import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def data_source_page():
    """数据源定义页面（修复：新增cleaned_df初始化 + card_style定义）"""
    st.header("数据导入与清洗配置")
    st.write("上传CSV文件后，配置数据清洗策略（实际清洗将在模型训练时基于训练集执行）。")

    # 初始化会话状态（核心修复：新增card_style定义，避免跨页面依赖）
    init_states = {
        "raw_df": None,  # 原始数据（不修改）
        "cleaned_df": None,  # 新增：初始化清洗后数据变量（即使未执行清洗也先定义）
        "cleaning_config": {},  # 清洗配置（用户选择的策略）
        "cleaning_steps": [],  # 清洗步骤记录（用于展示）
        "data_loaded": False,  # 数据是否已加载
        "all_features": [],  # 所有特征列
        "features": [],  # 初始特征列
        # 新增：定义card_style，与variable_definition.py保持一致
        "card_style": """
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        """
    }
    for key, val in init_states.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 1. 数据上传区域
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("1. 数据上传")

        uploaded_file = st.file_uploader(
            "上传CSV数据集",
            type="csv",
            key="file_uploader",
            help="支持格式：.csv，建议文件大小不超过100MB"
        )

        if uploaded_file is not None:
            try:
                with st.spinner("⏳ 正在加载数据..."):
                    # 自动尝试常见编码
                    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                    df = None
                    used_encoding = None

                    for encoding in encodings_to_try:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            used_encoding = encoding
                            break
                        except UnicodeDecodeError:
                            continue

                    if df is None:
                        st.error("❌ 无法解析文件，请检查文件格式或编码")
                        st.markdown("</div>", unsafe_allow_html=True)
                        return

                    # 保存原始数据 + 初始化cleaned_df（核心修复：避免cleaned_df为None）
                    st.session_state.raw_df = df.copy()
                    st.session_state.cleaned_df = df.copy()  # 初始化为原始数据，后续清洗时更新
                    st.session_state.data_loaded = True
                    st.session_state.all_features = df.columns.tolist()
                    st.session_state.features = df.columns.tolist()

                    # 计算原始数据质量指标（用于配置参考）
                    raw_duplicates = df.duplicated().sum()
                    raw_missing = df.isnull().sum().sum()
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    raw_outliers = {}
                    for col in numeric_cols:
                        z_scores = stats.zscore(df[col].dropna())
                        raw_outliers[col] = sum(np.abs(z_scores) > 3)  # 3σ法则

                    # 保存原始数据质量指标到配置
                    st.session_state.cleaning_config["raw_stats"] = {
                        "raw_duplicates": raw_duplicates,
                        "raw_missing": raw_missing,
                        "raw_outliers": raw_outliers,
                        "numeric_cols": numeric_cols,
                        "total_samples": len(df),
                        "total_features": len(df.columns)
                    }

                    st.success(
                        f"✅ 数据集加载成功！使用编码: {used_encoding}\n"
                        f"样本数: {len(df)}, 特征数: {len(df.columns)}, 重复样本: {raw_duplicates}, 缺失值: {raw_missing}"
                    )
            except Exception as e:
                st.error(f"❌ 数据加载错误: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. 原始数据质量分析（仅展示，不处理）
    if st.session_state.data_loaded and st.session_state.raw_df is not None:
        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("2. 原始数据质量分析")
            raw_stats = st.session_state.cleaning_config.get("raw_stats", {})
            numeric_cols = raw_stats.get("numeric_cols", [])

            # 质量指标卡片
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总样本数", raw_stats.get("total_samples", 0))
            with col2:
                st.metric("重复样本数", raw_stats.get("raw_duplicates", 0))
            with col3:
                st.metric("总缺失值", raw_stats.get("raw_missing", 0))
            with col4:
                st.metric("数值特征数", len(numeric_cols))

            # 缺失值分布可视化（前10个特征）
            st.subheader("缺失值分布（前10个特征）")
            missing_ratio = st.session_state.raw_df.isnull().mean().sort_values(ascending=False)[:10]
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_ratio.plot(kind='bar', ax=ax, color='salmon')
            ax.set_title("各特征缺失值比例（前10）")
            ax.set_ylabel("缺失比例")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # 数值特征异常值可视化（前10个特征）
            if len(numeric_cols) > 0:
                st.subheader("数值特征异常值分布（箱线图，前10个特征）")
                numeric_cols_display = numeric_cols[:10]
                n_cols = min(3, len(numeric_cols_display))
                n_rows = (len(numeric_cols_display) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

                for i, col in enumerate(numeric_cols_display):
                    sns.boxplot(x=st.session_state.raw_df[col], ax=axes[i], color='lightblue')
                    axes[i].set_title(f"{col}（异常值: {raw_stats.get('raw_outliers', {}).get(col, 0)}）")

                # 隐藏未使用的子图
                for i in range(len(numeric_cols_display), len(axes)):
                    axes[i].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. 清洗配置（核心：仅保存策略，不执行）
        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("3. 数据清洗策略配置")
            st.write("选择清洗规则，实际清洗将在模型训练时基于**训练集**执行（避免数据泄露）")
            raw_stats = st.session_state.cleaning_config.get("raw_stats", {})
            raw_duplicates = raw_stats.get("raw_duplicates", 0)
            raw_missing = raw_stats.get("raw_missing", 0)
            numeric_cols = raw_stats.get("numeric_cols", [])
            raw_outliers = raw_stats.get("raw_outliers", {})

            # 3.1 重复值处理
            with st.expander("重复值处理", expanded=True):
                if raw_duplicates > 0:
                    st.warning(f"检测到 {raw_duplicates} 条重复样本")
                    handle_duplicates = st.radio(
                        "处理方式:",
                        ["保留重复值", "删除重复值"],
                        index=1
                    )
                else:
                    st.success("未检测到重复样本")
                    handle_duplicates = "保留重复值"

            # 3.2 缺失值处理
            with st.expander("缺失值处理", expanded=True):
                if raw_missing > 0:
                    st.warning(f"检测到 {raw_missing} 个缺失值")
                    missing_handle_method = st.selectbox(
                        "处理方式:",
                        ["不处理", "删除含缺失值的行", "删除含缺失值的列", "填充缺失值"]
                    )

                    # 填充方式（仅当选择填充时显示）
                    fill_method = None
                    fill_value = None
                    if missing_handle_method == "填充缺失值":
                        fill_method = st.radio(
                            "填充策略:",
                            ["数值型用中位数/类别型用众数", "数值型用均值/类别型用众数", "自定义值填充"]
                        )
                        if fill_method == "自定义值填充":
                            fill_value = st.text_input("输入填充值", "0")
                else:
                    st.success("未检测到缺失值")
                    missing_handle_method = "不处理"
                    fill_method = None
                    fill_value = None

            # 3.3 异常值处理（仅对数值型特征）
            with st.expander("异常值处理", expanded=True):
                total_outliers = sum(raw_outliers.values()) if raw_outliers else 0
                if len(numeric_cols) > 0 and total_outliers > 0:
                    st.warning(f"检测到 {total_outliers} 个异常值（3σ法则）")
                    outlier_handle_method = st.selectbox(
                        "处理方式:",
                        ["不处理", "截断异常值（替换为临界值）", "删除含异常值的行"]
                    )
                else:
                    st.success("未检测到异常值")
                    outlier_handle_method = "不处理"

            # 保存配置按钮（核心：不执行清洗，仅存配置）
            if st.button("确认清洗策略", use_container_width=True, type="primary"):
                with st.spinner("⏳ 保存清洗策略..."):
                    # 保存清洗配置到会话状态
                    st.session_state.cleaning_config.update({
                        "handle_duplicates": handle_duplicates,
                        "missing_handle_method": missing_handle_method,
                        "fill_method": fill_method,
                        "fill_value": fill_value,
                        "outlier_handle_method": outlier_handle_method
                    })

                    # 生成清洗步骤记录（用于展示）
                    cleaning_steps = []
                    if handle_duplicates == "删除重复值" and raw_duplicates > 0:
                        cleaning_steps.append(f"1. 重复值处理：删除训练集中 {raw_duplicates} 条重复样本（测试集不处理）")
                    if missing_handle_method != "不处理" and raw_missing > 0:
                        if missing_handle_method == "删除含缺失值的行":
                            cleaning_steps.append(f"2. 缺失值处理：删除训练集含缺失值的行（测试集不处理）")
                        elif missing_handle_method == "删除含缺失值的列":
                            cleaning_steps.append(f"2. 缺失值处理：删除训练集含缺失值的列（同步应用到测试集）")
                        elif missing_handle_method == "填充缺失值":
                            if fill_method == "数值型用中位数/类别型用众数":
                                cleaning_steps.append(
                                    f"2. 缺失值处理：数值型用训练集中位数，类别型用训练集众数（测试集用相同统计量）")
                            elif fill_method == "数值型用均值/类别型用众数":
                                cleaning_steps.append(
                                    f"2. 缺失值处理：数值型用训练集均值，类别型用训练集众数（测试集用相同统计量）")
                            elif fill_method == "自定义值填充":
                                cleaning_steps.append(f"2. 缺失值处理：所有缺失值用 {fill_value} 填充（训练集+测试集）")
                    if outlier_handle_method != "不处理" and total_outliers > 0:
                        if outlier_handle_method == "截断异常值（替换为临界值）":
                            cleaning_steps.append(f"3. 异常值处理：用训练集3σ临界值截断（测试集用相同临界值）")
                        elif outlier_handle_method == "删除含异常值的行":
                            cleaning_steps.append(f"3. 异常值处理：删除训练集含异常值的行（测试集不处理）")

                    st.session_state.cleaning_steps = cleaning_steps
                    st.success("✅ 清洗策略已保存！实际清洗将在模型训练时执行")
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. 配置预览
        if st.session_state.cleaning_steps:
            with st.container():
                st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
                st.subheader("4. 清洗策略预览")
                st.write("以下步骤将在模型训练时基于**训练集**自动执行：")
                for step in st.session_state.cleaning_steps:
                    st.write(f"✅ {step}")
                st.markdown("</div>", unsafe_allow_html=True)

    # 5. 下一步按钮（修复：确保page_flow存在）
    # 初始化page_flow（避免未定义导致的错误）
    if "page_flow" not in st.session_state:
        st.session_state.page_flow = ["数据导入与预览", "变量定义与任务设置", "模型训练与配置"]

    page_flow = st.session_state.page_flow
    current_idx = page_flow.index(st.session_state.current_page) if st.session_state.current_page in page_flow else 0

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button(
                "下一步：变量定义与任务设置 →",
                use_container_width=True,
                type="primary",
                disabled=not st.session_state.data_loaded
        ):
            next_idx = current_idx + 1 if current_idx + 1 < len(page_flow) else current_idx
            st.session_state.current_page = page_flow[next_idx]
            st.rerun()