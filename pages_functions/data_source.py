import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def data_source_page():
    """数据源定义页面（修复索引错误并优化可视化）"""
    st.header("数据导入与清洗")
    st.write("上传CSV文件后，可进行数据清洗并实时查看清洗效果。")

    # 初始化清洗相关的会话状态
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None  # 原始数据
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None  # 清洗后数据
    if 'cleaning_steps' not in st.session_state:
        st.session_state.cleaning_steps = []  # 记录清洗步骤

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

                    # 保存原始数据和初始化清洗数据
                    st.session_state.raw_df = df.copy()
                    st.session_state.cleaned_df = df.copy()
                    st.session_state.data_loaded = True
                    st.session_state.all_features = df.columns.tolist()
                    st.session_state.features = df.columns.tolist()

                    st.success(
                        f"✅ 数据集加载成功！使用编码: {used_encoding}，"
                        f"样本数: {len(df)}, 特征数: {len(df.columns)}"
                    )
            except Exception as e:
                st.error(f"❌ 数据加载错误: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. 数据清洗区域（仅当数据加载后显示）
    if st.session_state.data_loaded and st.session_state.raw_df is not None:
        # 显示原始数据质量报告
        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("2. 原始数据质量分析")

            # 重复值统计
            raw_duplicates = st.session_state.raw_df.duplicated().sum()
            # 缺失值统计
            raw_missing = st.session_state.raw_df.isnull().sum().sum()
            # 异常值统计（针对数值型特征）
            numeric_cols = st.session_state.raw_df.select_dtypes(include=np.number).columns.tolist()
            # 限制显示的数值特征最多10个
            numeric_cols_display = numeric_cols[:10]
            raw_outliers = {}
            for col in numeric_cols:
                z_scores = stats.zscore(st.session_state.raw_df[col].dropna())
                raw_outliers[col] = sum(np.abs(z_scores) > 3)  # 3σ法则

            # 质量指标卡片
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总样本数", len(st.session_state.raw_df))
            with col2:
                st.metric("重复样本数", raw_duplicates)
            with col3:
                st.metric("总缺失值数量", raw_missing)

            # 缺失值分布可视化（只显示前10个特征）
            st.subheader("缺失值分布（前10个特征）")
            missing_ratio = st.session_state.raw_df.isnull().mean().sort_values(ascending=False)
            # 只取前10个特征
            missing_ratio_top10 = missing_ratio[:10]
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_ratio_top10.plot(kind='bar', ax=ax, color='salmon')
            ax.set_title("各特征缺失值比例（前10）")
            ax.set_ylabel("缺失比例")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # 数值特征异常值可视化（只显示前10个特征）
            if len(numeric_cols) > 0:
                st.subheader("数值特征异常值分布（箱线图，前10个特征）")
                # 最多显示10个特征
                numeric_cols_display = numeric_cols[:10]
                n_cols = min(3, len(numeric_cols_display))
                n_rows = (len(numeric_cols_display) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

                for i, col in enumerate(numeric_cols_display):
                    sns.boxplot(x=st.session_state.raw_df[col], ax=axes[i], color='lightblue')
                    axes[i].set_title(f"{col} (异常值: {raw_outliers[col]})")

                # 隐藏未使用的子图
                for i in range(len(numeric_cols_display), len(axes)):
                    axes[i].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        # 清洗步骤配置
        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("3. 数据清洗配置")
            st.write("选择清洗步骤，系统将实时展示清洗效果")

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
                    if missing_handle_method == "填充缺失值":
                        fill_method = st.radio(
                            "填充策略:",
                            ["数值型用中位数/类别型用众数", "数值型用均值/类别型用众数", "自定义值填充"]
                        )

                        # 自定义填充值（仅当选择自定义时显示）
                        if fill_method == "自定义值填充":
                            fill_value = st.text_input("输入填充值", "0")
                    else:
                        fill_method = None
                        fill_value = None
                else:
                    st.success("未检测到缺失值")
                    missing_handle_method = "不处理"
                    fill_method = None
                    fill_value = None

            # 3.3 异常值处理（仅对数值型特征）
            with st.expander("异常值处理", expanded=True):
                if len(numeric_cols) > 0 and sum(raw_outliers.values()) > 0:
                    st.warning(f"检测到 {sum(raw_outliers.values())} 个异常值（3σ法则）")
                    outlier_handle_method = st.selectbox(
                        "处理方式:",
                        ["不处理", "截断异常值（替换为临界值）", "删除含异常值的行"]
                    )
                else:
                    st.success("未检测到异常值")
                    outlier_handle_method = "不处理"

            # 执行清洗按钮
            if st.button("执行数据清洗", use_container_width=True, type="primary"):
                with st.spinner("⏳ 正在执行数据清洗..."):
                    # 基于原始数据重新清洗（避免多次清洗累积）
                    cleaned_df = st.session_state.raw_df.copy()
                    steps = []

                    # 处理重复值
                    if handle_duplicates == "删除重复值" and raw_duplicates > 0:
                        cleaned_df = cleaned_df.drop_duplicates()
                        steps.append(f"删除重复值: {raw_duplicates} 条重复样本")

                    # 处理缺失值
                    if missing_handle_method != "不处理" and raw_missing > 0:
                        if missing_handle_method == "删除含缺失值的行":
                            before = len(cleaned_df)
                            cleaned_df = cleaned_df.dropna(axis=0)
                            after = len(cleaned_df)
                            steps.append(f"删除含缺失值的行: 从 {before} 行减少到 {after} 行")
                        elif missing_handle_method == "删除含缺失值的列":
                            before = len(cleaned_df.columns)
                            cleaned_df = cleaned_df.dropna(axis=1)
                            after = len(cleaned_df.columns)
                            steps.append(f"删除含缺失值的列: 从 {before} 列减少到 {after} 列")
                        elif missing_handle_method == "填充缺失值":
                            numeric_cols_clean = cleaned_df.select_dtypes(include=np.number).columns.tolist()
                            cat_cols_clean = cleaned_df.select_dtypes(exclude=np.number).columns.tolist()

                            if fill_method == "数值型用中位数/类别型用众数":
                                # 填充数值型特征
                                cleaned_df[numeric_cols_clean] = cleaned_df[numeric_cols_clean].fillna(
                                    cleaned_df[numeric_cols_clean].median()
                                )

                                # 填充类别型特征（安全处理）
                                for col in cat_cols_clean:
                                    # 检查是否有众数
                                    mode_vals = cleaned_df[col].mode()
                                    if not mode_vals.empty:
                                        cleaned_df[col] = cleaned_df[col].fillna(mode_vals.iloc[0])
                                    else:
                                        # 如果没有众数，使用最常见的填充方式
                                        cleaned_df[col] = cleaned_df[col].fillna("未知")

                                steps.append("填充缺失值: 数值型用中位数，类别型用众数或'未知'")

                            elif fill_method == "数值型用均值/类别型用众数":
                                # 填充数值型特征
                                cleaned_df[numeric_cols_clean] = cleaned_df[numeric_cols_clean].fillna(
                                    cleaned_df[numeric_cols_clean].mean()
                                )

                                # 填充类别型特征（安全处理）
                                for col in cat_cols_clean:
                                    mode_vals = cleaned_df[col].mode()
                                    if not mode_vals.empty:
                                        cleaned_df[col] = cleaned_df[col].fillna(mode_vals.iloc[0])
                                    else:
                                        cleaned_df[col] = cleaned_df[col].fillna("未知")

                                steps.append("填充缺失值: 数值型用均值，类别型用众数或'未知'")

                            elif fill_method == "自定义值填充":
                                cleaned_df = cleaned_df.fillna(fill_value)
                                steps.append(f"填充缺失值: 所有缺失值替换为 {fill_value}")

                    # 处理异常值
                    if outlier_handle_method != "不处理" and len(numeric_cols) > 0:
                        # 只处理前10个数值特征
                        for col in numeric_cols[:10]:
                            if raw_outliers[col] == 0:
                                continue

                            # 计算3σ临界值
                            mean = cleaned_df[col].mean()
                            std = cleaned_df[col].std()
                            upper_limit = mean + 3 * std
                            lower_limit = mean - 3 * std

                            if outlier_handle_method == "截断异常值（替换为临界值）":
                                cleaned_df[col] = np.where(cleaned_df[col] > upper_limit, upper_limit, cleaned_df[col])
                                cleaned_df[col] = np.where(cleaned_df[col] < lower_limit, lower_limit, cleaned_df[col])
                                steps.append(f"截断异常值: {col}（替换为3σ临界值）")
                            elif outlier_handle_method == "删除含异常值的行":
                                before = len(cleaned_df)
                                cleaned_df = cleaned_df[
                                    (cleaned_df[col] >= lower_limit) & (cleaned_df[col] <= upper_limit)]
                                after = len(cleaned_df)
                                steps.append(f"删除异常值: {col}（从 {before} 行减少到 {after} 行）")

                    # 保存清洗结果
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.cleaning_steps = steps
                    st.session_state.df = cleaned_df  # 更新用于后续步骤的数据
                    st.session_state.all_features = cleaned_df.columns.tolist()
                    st.session_state.features = cleaned_df.columns.tolist()

                    st.success("✅ 数据清洗完成！")
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. 清洗结果可视化（仅当清洗后显示）
        if st.session_state.cleaned_df is not None and len(st.session_state.cleaning_steps) > 0:
            with st.container():
                st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
                st.subheader("4. 清洗结果对比")

                # 显示清洗步骤
                st.subheader("清洗步骤记录")
                for i, step in enumerate(st.session_state.cleaning_steps, 1):
                    st.write(f"{i}. {step}")

                # 清洗前后指标对比
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("清洗前")
                    st.dataframe({
                        "指标": ["样本数", "特征数", "重复样本数", "缺失值总数"],
                        "值": [
                            len(st.session_state.raw_df),
                            len(st.session_state.raw_df.columns),
                            st.session_state.raw_df.duplicated().sum(),
                            st.session_state.raw_df.isnull().sum().sum()
                        ]
                    }, use_container_width=True)
                with col2:
                    st.subheader("清洗后")
                    st.dataframe({
                        "指标": ["样本数", "特征数", "重复样本数", "缺失值总数"],
                        "值": [
                            len(st.session_state.cleaned_df),
                            len(st.session_state.cleaned_df.columns),
                            st.session_state.cleaned_df.duplicated().sum(),
                            st.session_state.cleaned_df.isnull().sum().sum()
                        ]
                    }, use_container_width=True)

                # 缺失值处理效果可视化（前10个特征）
                st.subheader("缺失值处理效果（前10个特征）")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                # 清洗前（前10）
                missing_ratio_raw = st.session_state.raw_df.isnull().mean().sort_values(ascending=False)[:10]
                missing_ratio_raw.plot(kind='bar', ax=ax1, color='salmon')
                ax1.set_title("清洗前缺失值比例")
                ax1.set_ylabel("缺失比例")
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

                # 清洗后（前10）
                missing_ratio_clean = st.session_state.cleaned_df.isnull().mean().sort_values(ascending=False)[:10]
                missing_ratio_clean.plot(kind='bar', ax=ax2, color='lightgreen')
                ax2.set_title("清洗后缺失值比例")
                ax2.set_ylabel("缺失比例")
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

                plt.tight_layout()
                st.pyplot(fig)

                # 异常值处理效果可视化（仅对数值型，前10个）
                if len(numeric_cols) > 0 and sum(raw_outliers.values()) > 0:
                    st.subheader("异常值处理效果（箱线图对比，前10个特征）")
                    # 最多显示10个特征
                    numeric_cols_display = numeric_cols[:10]
                    n_cols = min(3, len(numeric_cols_display))
                    n_rows = (len(numeric_cols_display) + n_cols - 1) // n_cols
                    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(10 * n_cols, 4 * n_rows))

                    for i, col in enumerate(numeric_cols_display):
                        row = i // n_cols
                        col_idx = (i % n_cols) * 2

                        # 清洗前
                        sns.boxplot(x=st.session_state.raw_df[col], ax=axes[row, col_idx], color='lightblue')
                        axes[row, col_idx].set_title(f"清洗前: {col}")

                        # 清洗后
                        sns.boxplot(x=st.session_state.cleaned_df[col], ax=axes[row, col_idx + 1], color='lightgreen')
                        axes[row, col_idx + 1].set_title(f"清洗后: {col}")

                    plt.tight_layout()
                    st.pyplot(fig)

                # 数据预览
                st.subheader("清洗后数据预览")
                st.dataframe(st.session_state.cleaned_df.head(10), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # 4. 下一步按钮
        page_flow = st.session_state.page_flow
        current_idx = page_flow.index(st.session_state.current_page)

        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button(
                    "下一步：变量定义与任务设置 →",
                    use_container_width=True,
                    type="primary",
                    disabled=not st.session_state.data_loaded
            ):
                st.session_state.current_page = page_flow[current_idx + 1]
                st.rerun()
