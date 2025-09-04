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


# 兼容处理：定义跨版本的rerun函数
def rerun():
    if hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.rerun()


def variable_definition_page():
    """变量定义与任务设置页面（修复标签页切换显示错误）"""
    st.header("变量定义与任务设置")

    # -------------------------- 初始化状态 --------------------------
    init_states = {
        "exclude_features": [],
        "manual_features": [],
        "auto_features": [],
        "confirmed_manual_features": [],
        "auto_select_params": {"method": None, "k": None, "lasso_alpha": 0.1},
        "active_selection_mode": "manual",
        "feature_selection_mode": "手动选择特征",
        "features": [],
        "manual_confirm_trigger": 0,
        # 新增：用于追踪标签页切换的状态
        "tab_just_changed": False,
        "card_style": """
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        """,
        "page_flow": ["数据源", "变量定义与任务设置", "模型训练与评估"]
    }
    for key, val in init_states.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # -------------------------- 前置检查 --------------------------
    if not st.session_state.get('data_loaded', False) or st.session_state.df is None:
        st.warning("⚠️ 请先在【数据源】页面上传数据！")
        if st.button("← 返回数据源", use_container_width=True):
            st.session_state.current_page = st.session_state.page_flow[0]
            rerun()
        return

    df = st.session_state.df
    all_features = [col for col in df.columns if col not in ['Unnamed: 0', 'index']]
    st.session_state.all_features = all_features

    # -------------------------- 基础配置 --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("1. 基础配置")

        # 任务类型
        task_type = st.radio(
            "任务类型",
            ["分类任务", "回归任务"],
            index=0 if st.session_state.get("task_type") != "回归任务" else 1,
            help="分类=预测类别，回归=预测连续值"
        )
        st.session_state.task_type = task_type

        # 目标变量
        default_target_idx = 0
        if st.session_state.get("target_col") in all_features:
            default_target_idx = all_features.index(st.session_state.target_col)
        target_col = st.selectbox(
            "目标变量（待预测列）",
            all_features,
            index=default_target_idx
        )
        st.session_state.target_col = target_col

        # 排除无关特征
        st.subheader("2. 排除无关特征（可选）")
        exclude_candidates = [f for f in all_features if f != target_col]
        if exclude_candidates:
            col_excl_all, col_excl_clear = st.columns(2)
            with col_excl_all:
                if st.button("全选排除", use_container_width=True):
                    st.session_state.exclude_features = exclude_candidates.copy()
                    st.session_state.manual_features = [
                        f for f in st.session_state.manual_features
                        if f not in exclude_candidates and f != target_col
                    ]
                    st.session_state.confirmed_manual_features = [
                        f for f in st.session_state.confirmed_manual_features
                        if f not in exclude_candidates and f != target_col
                    ]
                    st.session_state.auto_features = [
                        f for f in st.session_state.auto_features
                        if f not in exclude_candidates and f != target_col
                    ]
                    rerun()
            with col_excl_clear:
                if st.button("取消所有排除", use_container_width=True):
                    st.session_state.exclude_features = []
                    rerun()

            excl_cols = st.columns(min(3, len(exclude_candidates)))
            for idx, feat in enumerate(exclude_candidates):
                with excl_cols[idx % 3]:
                    is_excluded = feat in st.session_state.exclude_features
                    if st.checkbox(feat, value=is_excluded, key=f"exclude_{feat}"):
                        if feat not in st.session_state.exclude_features:
                            st.session_state.exclude_features.append(feat)
                            if feat in st.session_state.manual_features:
                                st.session_state.manual_features.remove(feat)
                            if feat in st.session_state.confirmed_manual_features:
                                st.session_state.confirmed_manual_features.remove(feat)
                            if feat in st.session_state.auto_features:
                                st.session_state.auto_features.remove(feat)
                    else:
                        if feat in st.session_state.exclude_features:
                            st.session_state.exclude_features.remove(feat)

            if st.session_state.exclude_features:
                st.warning(
                    f"已排除 {len(st.session_state.exclude_features)} 个特征：{', '.join(st.session_state.exclude_features)}")
        else:
            st.info("无特征可排除（仅剩余目标变量）")

        # 计算候选特征
        candidate_features = [
            f for f in all_features
            if f != target_col and f not in st.session_state.exclude_features
        ]
        st.session_state.candidate_features = candidate_features

        # 首次初始化
        if not st.session_state.manual_features and candidate_features:
            st.session_state.manual_features = candidate_features.copy()
            st.session_state.confirmed_manual_features = candidate_features.copy()

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- 特征筛选 --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("3. 特征筛选（两种方式独立）")
        st.info(f"候选特征：{len(candidate_features)} 个（已排除目标+{len(st.session_state.exclude_features)}个无关特征）")

        # 核心修复：改进标签页切换检测机制
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "📝 手动筛选"

        # 标签页切换回调函数（核心修复：添加立即刷新逻辑）
        def set_tab(tab_name):
            if st.session_state.current_tab != tab_name:
                st.session_state.current_tab = tab_name
                st.session_state.tab_just_changed = True  # 标记为刚切换

                # 更新激活模式
                if tab_name == "📝 手动筛选":
                    st.session_state.active_selection_mode = "manual"
                    st.session_state.feature_selection_mode = "手动选择特征"
                else:
                    st.session_state.active_selection_mode = "auto"
                    st.session_state.feature_selection_mode = "自动筛选特征"

                rerun()  # 切换后立即刷新

        # 创建标签页并绑定回调
        tab1, tab2 = st.tabs(["📝 手动筛选", "🔍 自动筛选"])

        # -------------------------- 手动筛选 --------------------------
        with tab1:
            # 隐性按钮用于触发切换检测
            st.button("手动筛选", on_click=set_tab, args=("📝 手动筛选",),
                      disabled=True, key="manual_indicator", help="当前激活的筛选方式")
            st.success("当前激活：手动筛选特征")

            st.write("从候选特征中手动选择，完成后请点击【确认手动选择】按钮")

            if candidate_features:
                # 清理无效特征
                st.session_state.manual_features = [
                    f for f in st.session_state.manual_features
                    if f in candidate_features
                ]

                man_cols = st.columns(min(3, len(candidate_features)))
                for idx, feat in enumerate(candidate_features):
                    with man_cols[idx % 3]:
                        is_selected = feat in st.session_state.manual_features

                        def update_selection(feature, selected):
                            if selected and feature not in st.session_state.manual_features:
                                st.session_state.manual_features.append(feature)
                            elif not selected and feature in st.session_state.manual_features:
                                st.session_state.manual_features.remove(feature)

                        st.checkbox(
                            feat,
                            value=is_selected,
                            key=f"manual_{feat}",
                            on_change=update_selection,
                            args=(feat, not is_selected)
                        )
            else:
                st.warning("无候选特征可选择")

            # 手动筛选确认按钮
            col_confirm_manual, _ = st.columns([1, 2])
            with col_confirm_manual:
                if st.button("✅ 确认手动选择", use_container_width=True, type="primary"):
                    valid_features = [f for f in st.session_state.manual_features if f in candidate_features]

                    if len(valid_features) == 0:
                        st.error("请至少选择1个特征后再确认")
                    else:
                        st.session_state.confirmed_manual_features = valid_features.copy()
                        st.success(f"已确认！手动筛选特征共 {len(valid_features)} 个")
                        rerun()

            # 显示确认状态
            valid_confirmed = [f for f in st.session_state.confirmed_manual_features if f in candidate_features]
            if valid_confirmed:
                st.info(f"当前已确认的手动特征：{len(valid_confirmed)} 个")
                if st.checkbox("预览已确认的手动特征", value=False):
                    st.dataframe(df[valid_confirmed].head(), use_container_width=True)
                    with st.expander("查看特征详情"):
                        st.write(", ".join(valid_confirmed))
            else:
                st.warning("尚未确认手动特征选择，请完成选择后点击【确认手动选择】按钮")

        # -------------------------- 自动筛选 --------------------------
        with tab2:
            # 隐性按钮用于触发切换检测
            st.button("自动筛选", on_click=set_tab, args=("🔍 自动筛选",),
                      disabled=True, key="auto_indicator", help="当前激活的筛选方式")
            st.success("当前激活：自动筛选特征")

            st.write("基于统计方法自动筛选，完成参数设置后请点击【执行自动筛选】按钮")

            if not candidate_features:
                st.warning("无候选特征可筛选")
            else:
                # 自动筛选参数
                col1, col2 = st.columns(2)
                with col1:
                    k = st.slider(
                        "保留特征数量",
                        min_value=1,
                        max_value=len(candidate_features),
                        value=min(5, len(candidate_features)),
                        key="auto_k"
                    )
                    st.session_state.auto_select_params["k"] = k

                with col2:
                    if task_type == "分类任务":
                        method = st.selectbox(
                            "筛选方法",
                            ["互信息（mutual_info_classif）", "卡方检验（chi2）", "Lasso回归（L1）"],
                            key="auto_method_clf"
                        )
                    else:
                        method = st.selectbox(
                            "筛选方法",
                            ["互信息（mutual_info_regression）", "皮尔逊相关（f_regression）", "Lasso回归（L1）"],
                            key="auto_method_reg"
                        )
                    st.session_state.auto_select_params["method"] = method

                # Lasso参数
                lasso_alpha = 0.1
                if "Lasso" in method:
                    lasso_alpha = st.slider(
                        "Lasso正则化强度（alpha）",
                        0.001, 1.0, 0.1, step=0.001,
                        key="auto_lasso_alpha"
                    )
                    st.session_state.auto_select_params["lasso_alpha"] = lasso_alpha

                # 执行自动筛选
                if st.button("🚀 执行自动筛选", use_container_width=True, type="primary"):
                    with st.spinner("筛选中..."):
                        try:
                            X = df[candidate_features].copy().fillna(df[candidate_features].median(numeric_only=True))
                            y = df[target_col].copy()
                            if task_type == "分类任务" and y.dtype == "object":
                                y = pd.factorize(y)[0]

                            X_scaled = X.copy()
                            if "Lasso" in method:
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X_scaled)

                            # 筛选逻辑
                            selected_features = []
                            scores = []
                            if "互信息" in method:
                                selector = SelectKBest(
                                    mutual_info_classif if task_type == "分类任务" else mutual_info_regression,
                                    k=k
                                )
                                selector.fit(X_scaled, y)
                                selected_mask = selector.get_support()
                                selected_features = [f for f, mask in zip(candidate_features, selected_mask) if mask]
                                scores = selector.scores_

                            elif "卡方" in method:
                                X_pos = X_scaled - X_scaled.min() + 1e-6
                                selector = SelectKBest(chi2, k=k)
                                selector.fit(X_pos, y)
                                selected_mask = selector.get_support()
                                selected_features = [f for f, mask in zip(candidate_features, selected_mask) if mask]
                                scores = selector.scores_

                            elif "皮尔逊" in method:
                                selector = SelectKBest(f_regression, k=k)
                                selector.fit(X_scaled, y)
                                selected_mask = selector.get_support()
                                selected_features = [f for f, mask in zip(candidate_features, selected_mask) if mask]
                                scores = selector.scores_

                            elif "Lasso" in method:
                                model = LassoCV(alphas=[lasso_alpha], cv=5) if task_type == "分类任务" else Lasso(
                                    alpha=lasso_alpha)
                                model.fit(X_scaled, y)
                                coefs = np.abs(model.coef_)
                                top_idx = np.argsort(coefs)[-k:] if sum(coefs > 0) >= k else np.where(coefs > 1e-8)[0]
                                selected_features = [candidate_features[i] for i in top_idx]
                                scores = coefs

                            st.session_state.auto_features = selected_features.copy()
                            st.success(f"自动筛选完成！保留 {len(selected_features)} 个特征")
                            st.dataframe(
                                pd.DataFrame({"特征": candidate_features, "重要性": scores})
                                .sort_values("重要性", ascending=False)
                                .head(10),
                                use_container_width=True
                            )

                            top_feat = pd.DataFrame({"特征": candidate_features, "重要性": scores}).sort_values(
                                "重要性", ascending=False).head(10)
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.barplot(x="重要性", y="特征", data=top_feat, ax=ax, palette="viridis")
                            st.pyplot(fig)
                            rerun()

                        except Exception as e:
                            st.error(f"筛选失败：{str(e)}")

            # 自动筛选结果
            if st.session_state.auto_features:
                st.info(f"自动筛选已完成：{len(st.session_state.auto_features)} 个特征")
                if st.checkbox("预览自动筛选特征", value=False):
                    st.dataframe(df[st.session_state.auto_features].head(), use_container_width=True)
            else:
                st.warning("尚未执行自动筛选，请设置参数后点击【执行自动筛选】按钮")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- 训练参数+导航 --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("4. 训练参数设置")

        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.1)
        st.session_state.test_size = test_size

        random_state = st.number_input("随机种子", 0, 1000, 42)
        st.session_state.random_state = random_state

        st.markdown("</div>", unsafe_allow_html=True)

    # 导航按钮（核心修复：确保使用最新的标签页状态）
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("← 返回数据源", use_container_width=True):
            st.session_state.current_page = st.session_state.page_flow[0]
            rerun()

    with col2:
        # 核心修复：直接使用current_tab判断当前模式，确保与标签页同步
        if st.session_state.current_tab == "📝 手动筛选":
            final_features = [f for f in st.session_state.confirmed_manual_features if f in candidate_features]
            mode_text = "手动选择特征"
            valid = len(final_features) > 0
        else:
            final_features = st.session_state.auto_features
            mode_text = "自动筛选特征"
            valid = len(final_features) > 0

        # 核心修复：标签页刚切换时显示提示
        if st.session_state.tab_just_changed:
            st.session_state.tab_just_changed = False  # 重置标记
            st.info(f"已切换到：{mode_text}")

        # 显示当前状态
        if valid:
            st.info(f"即将使用：{mode_text}（{len(final_features)}个特征）")
            with st.expander("点击查看特征列表"):
                st.write(", ".join(final_features))
        else:
            if st.session_state.current_tab == "📝 手动筛选":
                st.warning("请先在手动筛选标签中完成特征选择并点击【确认手动选择】按钮")
            else:
                st.warning("请先在自动筛选标签中设置参数并点击【执行自动筛选】按钮")

        if st.button("下一步：模型训练与评估 →", use_container_width=True, type="primary", disabled=not valid):
            st.session_state.features = final_features.copy()
            st.session_state.feature_selection_mode = mode_text
            st.session_state.current_page = st.session_state.page_flow[2]
            st.success(f"已确认使用 {mode_text}，共 {len(final_features)} 个特征")
            rerun()
