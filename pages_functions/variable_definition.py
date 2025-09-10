# variable_definition_page.py
# 修复：1) 移除非循环中的continue 2) 增强数据检查 3) 兼容card_style初始化
# 2025-09-04 最终修复版

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    SelectKBest, chi2, f_regression
)
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler


# -------------------------- 工具函数 --------------------------
def rerun():
    """兼容 Streamlit 版本差异的 rerun 封装"""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()


def convert_boolean_strings(df):
    """将数据框中的TRUE/FALSE字符串列转换为数值(1/0)"""
    converted_df = df.copy()
    for col in converted_df.columns:
        # 仅处理object类型列
        if converted_df[col].dtype == 'object':
            # 获取非空唯一值并统一转为小写
            unique_vals = [str(v).lower() for v in converted_df[col].dropna().unique()]
            # 判断是否为布尔字符串类型（仅包含true/false）
            if set(unique_vals).issubset({'true', 'false'}):
                # 转换为1/0
                converted_df[col] = converted_df[col].str.lower().map({'true': 1, 'false': 0})
                # 转换为float类型避免后续警告
                converted_df[col] = converted_df[col].astype(float)
    return converted_df


# -------------------------- 主页面函数 --------------------------
def variable_definition_page():
    """变量定义与特征选择页面（已修复数据检查逻辑 & 移除continue错误）"""
    st.header("变量定义与任务设置")

    # -------------------------- 初始化状态 --------------------------
    init_states = {
        "exclude_features": [],
        "manual_features": [],
        "auto_features": [],
        "confirmed_manual_features": [],
        "auto_features_executed": [],  # 保存自动筛选结果
        "auto_feature_scores": [],  # 保存特征重要性分数
        "auto_select_params": {"method": None, "k": None, "lasso_alpha": 0.1},
        "active_selection_mode": "manual",  # manual / auto
        "feature_selection_mode": "手动选择特征",
        "features": [],
        "manual_confirm_trigger": 0,
        "current_tab": "手动筛选",
        # 定义card_style（与data_source.py保持一致，防止未定义）
        "card_style": """
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        """,
        "page_flow": ["数据导入与预览", "变量定义与任务设置", "模型训练与配置"]  # 与主流程保持一致
    }
    for k, v in init_states.items():
        st.session_state.setdefault(k, v)

    # -------------------------- 核心修复：前置检查（兼容清洗后数据） --------------------------
    # 1. 检查所有可能的数据存储变量（raw_df/cleaned_df）
    data_vars = {
        "cleaned_df": st.session_state.get("cleaned_df"),  # 清洗后的数据（优先使用）
        "raw_df": st.session_state.get("raw_df"),  # 原始数据（备用）
        "df": st.session_state.get("df")  # 旧数据变量（兼容）
    }
    valid_data = None
    data_source = None

    # 2. 寻找有效的数据（优先级：cleaned_df > raw_df > df）
    for var_name, data in data_vars.items():
        if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
            valid_data = data
            data_source = var_name
            break

    # 3. 检查数据加载状态（兼容旧的data_loaded和新的清洗状态）
    data_loaded = st.session_state.get("data_loaded", False)
    # 即使data_loaded未置为True，只要有有效数据也视为已加载
    if valid_data is not None:
        data_loaded = True
        st.session_state.data_loaded = True  # 同步更新状态
        st.session_state.df = valid_data  # 赋值给旧的df变量，确保后续逻辑兼容

    # 4. 数据未加载时的提示（增加诊断信息）
    if not data_loaded or valid_data is None:
        st.warning("⚠️ 请先在【数据导入与预览】页面上传数据并确认清洗策略！")

        # 数据状态诊断面板（帮助定位问题）
        with st.expander("🔍 数据状态诊断（点击展开）", expanded=False):
            st.write("### 数据变量检查")
            for var_name, data in data_vars.items():
                if data is None:
                    st.write(f"- **{var_name}**: ❌ 未定义（None）")
                elif not isinstance(data, pd.DataFrame):
                    st.write(f"- **{var_name}**: ❌ 类型错误（不是DataFrame）")
                elif data.empty:
                    st.write(f"- **{var_name}**: ❌ 数据为空（0行）")
                else:
                    st.write(f"- **{var_name}**: ✅ 有效（{data.shape[0]}行 × {data.shape[1]}列）")

            st.write("\n### 状态变量检查")
            st.write(f"- **data_loaded**: {'✅ True' if data_loaded else '❌ False'}")
            st.write(f"- **当前页面**: {st.session_state.get('current_page', '未定义')}")
            st.write(f"- **页面流程**: {st.session_state.get('page_flow', '未定义')}")

        # 返回数据源页面的按钮
        if st.button("← 返回数据导入与预览", use_container_width=True):
            st.session_state.current_page = st.session_state.page_flow[0]
            rerun()
        return

    # -------------------------- 基础配置（使用有效数据） --------------------------
    df = valid_data  # 使用诊断后的有效数据
    all_features = [c for c in df.columns if c not in {"Unnamed: 0", "index"}]
    st.session_state.all_features = all_features

    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("1. 基础配置")

        # 任务类型选择（兼容旧的task_type状态）
        task_type = st.radio(
            "任务类型",
            ["分类任务", "回归任务"],
            index=0 if st.session_state.get("task_type") != "回归任务" else 1,
            help="分类=预测类别（如疾病是否阳性），回归=预测连续值（如血糖值）"
        )
        st.session_state.task_type = task_type

        # 目标变量选择（兼容旧的target_col状态）
        default_target_idx = 0
        if st.session_state.get("target_col") in all_features:
            default_target_idx = all_features.index(st.session_state.target_col)
        # 防止索引越界（当特征列表变化时）
        default_target_idx = min(default_target_idx, len(all_features) - 1) if len(all_features) > 0 else 0

        target_col = st.selectbox(
            "目标变量（待预测列）",
            all_features,
            index=default_target_idx,
            help="选择你想要通过模型预测的列（如：疾病诊断结果、血压值）"
        )
        st.session_state.target_col = target_col

        # 排除无关特征（可选）
        st.subheader("2. 排除无关特征（可选）")
        exclude_candidates = [f for f in all_features if f != target_col]
        if exclude_candidates:
            # 全选/取消全选按钮
            col_select_all, col_clear_all = st.columns(2)
            with col_select_all:
                if st.button("全选排除", use_container_width=True):
                    st.session_state.exclude_features = exclude_candidates.copy()
                    # 同步清理已选特征列表
                    for lst in ["manual_features", "confirmed_manual_features", "auto_features",
                                "auto_features_executed"]:
                        st.session_state[lst] = [f for f in st.session_state[lst] if f not in exclude_candidates]
                    rerun()
            with col_clear_all:
                if st.button("取消所有排除", use_container_width=True):
                    st.session_state.exclude_features.clear()
                    rerun()

            # 特征排除复选框（按列排列）
            cols = st.columns(min(3, len(exclude_candidates)))
            for idx, feat in enumerate(exclude_candidates):
                with cols[idx % 3]:
                    checked = feat in st.session_state.exclude_features
                    # 复选框状态变化时同步更新排除列表
                    if st.checkbox(feat, value=checked, key=f"exclude_{feat}"):
                        if feat not in st.session_state.exclude_features:
                            st.session_state.exclude_features.append(feat)
                            # 从已选特征中移除
                            for lst in ["manual_features", "confirmed_manual_features", "auto_features",
                                        "auto_features_executed"]:
                                if feat in st.session_state[lst]:
                                    st.session_state[lst].remove(feat)
                    else:
                        if feat in st.session_state.exclude_features:
                            st.session_state.exclude_features.remove(feat)

        # 计算候选特征（排除目标变量和已选排除的特征）
        candidate_features = [
            f for f in all_features
            if f != target_col and f not in st.session_state.exclude_features
        ]
        st.session_state.candidate_features = candidate_features

        # 初始化手动特征列表（如果为空）
        if not st.session_state.manual_features and candidate_features:
            st.session_state.manual_features = candidate_features.copy()
            st.session_state.confirmed_manual_features = candidate_features.copy()

        # 显示候选特征数量
        st.info(
            f"✅ 候选特征数量：{len(candidate_features)} 个（已排除：目标变量×1 + 无关特征×{len(st.session_state.exclude_features)}）")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- 特征筛选（修复continue语法错误） --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("3. 特征筛选（两种方式独立）")
        st.info(f"候选特征：{len(candidate_features)} 个（已排除目标+{len(st.session_state.exclude_features)}个无关特征）")

        tab1, tab2 = st.tabs(["手动筛选", "自动筛选"])

        # 手动筛选
        with tab1:
            st.success("当前激活：手动筛选特征")
            if candidate_features:
                # 清理无效特征（确保已排除的特征不显示）
                st.session_state.manual_features = [
                    f for f in st.session_state.manual_features if f in candidate_features
                ]
                # 按列显示复选框
                man_cols = st.columns(min(3, len(candidate_features)))
                for idx, feat in enumerate(candidate_features):
                    with man_cols[idx % 3]:
                        sel = feat in st.session_state.manual_features

                        # 特征选择切换函数
                        def toggle(f=feat, s=sel):
                            if s:
                                if f in st.session_state.manual_features:
                                    st.session_state.manual_features.remove(f)
                            else:
                                st.session_state.manual_features.append(f)

                        st.checkbox(feat, value=sel, key=f"manual_{feat}", on_change=toggle)

            # 确认手动选择按钮
            col_confirm, _ = st.columns([1, 2])
            with col_confirm:
                if st.button("✅ 确认手动选择", use_container_width=True, type="primary"):
                    valid = [f for f in st.session_state.manual_features if f in candidate_features]
                    if not valid:
                        st.error("❌ 请至少选择1个特征后再确认")
                    else:
                        st.session_state.confirmed_manual_features = valid.copy()
                        st.session_state.active_selection_mode = "manual"
                        st.session_state.feature_selection_mode = "手动选择特征"
                        st.session_state.features = valid.copy()  # 同步到全局特征列表
                        st.success(f"✅ 已确认 {len(valid)} 个手动特征")
                        rerun()

            # 显示已确认的手动特征
            valid_confirmed = [f for f in st.session_state.confirmed_manual_features if f in candidate_features]
            if valid_confirmed:
                st.info(f"当前已确认的手动特征：{len(valid_confirmed)} 个")
                with st.expander("点击查看特征列表", expanded=False):
                    st.write(", ".join(valid_confirmed))

        # 自动筛选（增加特征重要性可视化）
        with tab2:
            st.success("当前激活：自动筛选特征")
            if not candidate_features:
                st.warning("⚠️ 无候选特征可筛选，请先在左侧排除无关特征后重试")
            else:
                # 自动筛选参数配置
                col1, col2 = st.columns(2)
                with col1:
                    # 保留特征数量（限制在1到候选特征数之间）
                    k = st.slider(
                        "保留特征数量",
                        min_value=1,
                        max_value=len(candidate_features),
                        value=min(5, len(candidate_features)),
                        key="auto_k",
                        help=f"最多可保留 {len(candidate_features)} 个特征"
                    )
                    st.session_state.auto_select_params["k"] = k
                with col2:
                    # 筛选方法选择（按任务类型区分）
                    if task_type == "分类任务":
                        method = st.selectbox(
                            "筛选方法",
                            ["互信息（mutual_info_classif）", "卡方检验（chi2）", "Lasso回归（L1）"],
                            key="auto_method_clf",
                            help="互信息：适合任何分类任务；卡方检验：适合非负特征；Lasso：适合高维数据"
                        )
                    else:
                        method = st.selectbox(
                            "筛选方法",
                            ["互信息（mutual_info_regression）", "皮尔逊相关（f_regression）", "Lasso回归（L1）"],
                            key="auto_method_reg",
                            help="互信息：适合任何回归任务；皮尔逊相关：适合线性关系；Lasso：适合高维数据"
                        )
                    st.session_state.auto_select_params["method"] = method

                # Lasso正则化参数（仅当选择Lasso方法时显示）
                lasso_alpha = 0.1
                if "Lasso" in method:
                    lasso_alpha = st.slider(
                        "Lasso正则化强度（alpha）",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.1,
                        step=0.001,
                        key="auto_lasso_alpha",
                        help="alpha越大，筛选出的特征越少（推荐0.01-0.1）"
                    )
                    st.session_state.auto_select_params["lasso_alpha"] = lasso_alpha

                # 执行自动筛选按钮
                if st.button("🔍 执行自动筛选", use_container_width=True, type="primary"):
                    with st.spinner("⏳ 正在执行特征筛选...（请耐心等待）"):
                        # 准备特征数据（包含TRUE/FALSE转换）
                        X = df[candidate_features].copy()

                        # 关键修复：转换TRUE/FALSE字符串为数值
                        X = convert_boolean_strings(X)

                        # 处理缺失值（仅对数值型特征用中位数填充）
                        numeric_cols = X.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                            st.info(f"ℹ️ 已用中位数填充 {len(numeric_cols)} 个数值型特征的缺失值")

                        # 处理非数值型特征（编码为数值）
                        object_cols = X.select_dtypes(include=['object']).columns
                        if len(object_cols) > 0:
                            for col in object_cols:
                                # 对字符串特征进行简单编码（适用于低基数特征）
                                X[col] = pd.factorize(X[col])[0]
                            st.info(f"ℹ️ 已编码 {len(object_cols)} 个非数值型特征（字符串→数值）")

                        # 准备目标变量
                        y = df[target_col].copy()
                        # 处理目标变量的缺失值（删除含缺失值的行）
                        if y.isnull().sum() > 0:
                            drop_rows = y.isnull()
                            X = X[~drop_rows]
                            y = y[~drop_rows]
                            st.info(f"ℹ️ 已删除目标变量中 {drop_rows.sum()} 个含缺失值的样本")

                        # 检查处理后的数据是否有效（核心修复：用if-else替代continue）
                        if X.empty or y.empty:
                            st.error("❌ 数据处理后为空，请检查原始数据质量（如缺失值过多、特征数量不足）")
                        else:
                            # 根据任务类型和方法选择特征筛选器
                            try:
                                selected_indices = []
                                scores = []
                                if task_type == "分类任务":
                                    if method == "互信息（mutual_info_classif）":
                                        selector = SelectKBest(mutual_info_classif, k=k)
                                        X_selected = selector.fit_transform(X, y)
                                        scores = selector.scores_
                                        selected_indices = selector.get_support(indices=True)
                                    elif method == "卡方检验（chi2）":
                                        # 卡方检验要求非负特征，这里做简单处理（平移到非负）
                                        X_pos = X - X.min() if X.min().min() < 0 else X
                                        selector = SelectKBest(chi2, k=k)
                                        X_selected = selector.fit_transform(X_pos, y)
                                        scores = selector.scores_
                                        selected_indices = selector.get_support(indices=True)
                                    elif method == "Lasso回归（L1）":
                                        # Lasso需要标准化特征
                                        scaler = StandardScaler()
                                        X_scaled = scaler.fit_transform(X)
                                        # 使用Lasso筛选特征（保留系数非零的特征）
                                        lasso = Lasso(alpha=lasso_alpha, max_iter=10000, random_state=42)
                                        lasso.fit(X_scaled, y)
                                        coefs = np.abs(lasso.coef_)
                                        scores = coefs
                                        # 按系数排序取前k个特征
                                        selected_indices = np.argsort(coefs)[-k:][::-1]
                                else:  # 回归任务
                                    if method == "互信息（mutual_info_regression）":
                                        selector = SelectKBest(mutual_info_regression, k=k)
                                        X_selected = selector.fit_transform(X, y)
                                        scores = selector.scores_
                                        selected_indices = selector.get_support(indices=True)
                                    elif method == "皮尔逊相关（f_regression）":
                                        selector = SelectKBest(f_regression, k=k)
                                        X_selected = selector.fit_transform(X, y)
                                        scores = selector.scores_
                                        selected_indices = selector.get_support(indices=True)
                                    elif method == "Lasso回归（L1）":
                                        scaler = StandardScaler()
                                        X_scaled = scaler.fit_transform(X)
                                        lasso = Lasso(alpha=lasso_alpha, max_iter=10000, random_state=42)
                                        lasso.fit(X_scaled, y)
                                        coefs = np.abs(lasso.coef_)
                                        scores = coefs
                                        selected_indices = np.argsort(coefs)[-k:][::-1]

                                # 提取筛选后的特征
                                selected_features = [candidate_features[i] for i in selected_indices]
                                # 确保特征列表有效（过滤掉不在候选特征中的值）
                                selected_features = [f for f in selected_features if f in candidate_features]
                                if not selected_features:
                                    st.warning("⚠️ 筛选后未获得有效特征，将使用前5个候选特征")
                                    selected_features = candidate_features[:min(5, len(candidate_features))]

                                # 保存自动筛选结果到会话状态
                                st.session_state.auto_features_executed = selected_features.copy()
                                st.session_state.features = selected_features.copy()  # 同步到全局特征列表
                                st.session_state.active_selection_mode = "auto"
                                st.session_state.feature_selection_mode = "自动筛选特征"

                                # 保存特征分数用于可视化（过滤无效分数）
                                feature_scores = {candidate_features[i]: scores[i] for i in
                                                  range(len(candidate_features))}
                                feature_scores = {k: v for k, v in feature_scores.items() if
                                                  not (np.isnan(v) or np.isinf(v))}
                                st.session_state.auto_feature_scores = sorted(feature_scores.items(),
                                                                              key=lambda x: x[1], reverse=True)

                                # 显示筛选结果
                                st.success(f"✅ 自动筛选完成，保留 {len(selected_features)} 个特征")
                                with st.expander("查看筛选结果", expanded=True):
                                    st.write("**筛选后的特征列表**：", ", ".join(selected_features))
                                    st.write(f"**筛选方法**：{method}")
                                    st.write(f"**保留特征数**：{len(selected_features)}")

                                # 特征重要性可视化（仅当有有效分数时）
                                if st.session_state.auto_feature_scores:
                                    top_n = min(15, len(st.session_state.auto_feature_scores))
                                    top_features = [x[0] for x in st.session_state.auto_feature_scores[:top_n]]
                                    top_scores = [x[1] for x in st.session_state.auto_feature_scores[:top_n]]

                                    # 绘制特征重要性图
                                    plt.figure(figsize=(10, 6))
                                    sns.barplot(x=top_scores, y=top_features, palette="viridis")
                                    plt.title(f"特征重要性分数（Top {top_n}）- 方法：{method}", fontsize=12)
                                    plt.xlabel("重要性分数", fontsize=10)
                                    plt.ylabel("特征名称", fontsize=10)
                                    plt.tight_layout()
                                    st.pyplot(plt)

                            except Exception as e:
                                st.error(f"❌ 筛选失败：{str(e)}")
                                st.exception(e)  # 显示详细错误信息（便于调试）

            # 显示自动筛选结果（如果已执行）
            if st.session_state.auto_features_executed:
                st.info(f"已筛选的自动特征：{len(st.session_state.auto_features_executed)} 个")
                with st.expander("点击查看特征列表", expanded=False):
                    st.write(", ".join(st.session_state.auto_features_executed))

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- 导航按钮（修复页面流程索引） --------------------------
    with st.container():
        # 确保页面流程正确
        page_flow = st.session_state.page_flow
        # 定位当前页面索引（防止索引越界）
        try:
            current_idx = page_flow.index(st.session_state.current_page)
        except ValueError:
            current_idx = 1  # 默认索引（变量定义页面）

        col_prev, col_next = st.columns(2)
        with col_prev:
            # 上一步：仅当不是第一个页面时显示
            if current_idx > 0:
                if st.button(f"← 上一步：{page_flow[current_idx - 1]}", use_container_width=True):
                    st.session_state.current_page = page_flow[current_idx - 1]
                    rerun()

        with col_next:
            # 检查是否已选择特征（手动/自动）
            has_valid_features = False
            if st.session_state.active_selection_mode == "manual":
                has_valid_features = len(st.session_state.confirmed_manual_features) > 0
            elif st.session_state.active_selection_mode == "auto":
                has_valid_features = len(st.session_state.auto_features_executed) > 0

            # 下一步按钮：仅当有有效特征时启用
            if current_idx < len(page_flow) - 1:
                if st.button(
                        f"下一步：{page_flow[current_idx + 1]} →",
                        use_container_width=True,
                        type="primary",
                        disabled=not has_valid_features
                ):
                    st.session_state.current_page = page_flow[current_idx + 1]
                    rerun()
                # 显示禁用原因
                if not has_valid_features:
                    if st.session_state.active_selection_mode == "manual":
                        st.caption("⚠️ 请先在「手动筛选」标签页选择特征并点击「确认手动选择」")
                    else:
                        st.caption("⚠️ 请先在「自动筛选」标签页点击「执行自动筛选」")