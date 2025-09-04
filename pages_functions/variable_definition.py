# variable_definition_page.py
# 修复：1) 自动筛选结果丢失  2) 导航按钮仍使用手动特征
# 新增：自动筛选特征重要性可视化
# 2025-09-04  完整修复版

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


# -------------------------- 主页面函数 --------------------------
def variable_definition_page():
    """变量定义与特征选择页面（已修复自动筛选结果丢失 & 导航误判，增加可视化）"""
    st.header("变量定义与任务设置")

    # -------------------------- 初始化状态 --------------------------
    init_states = {
        "exclude_features": [],
        "manual_features": [],
        "auto_features": [],
        "confirmed_manual_features": [],
        "auto_features_executed": [],  # 保存自动筛选结果
        "auto_feature_scores": [],  # 新增：保存特征重要性分数
        "auto_select_params": {"method": None, "k": None, "lasso_alpha": 0.1},
        "active_selection_mode": "manual",  # manual / auto
        "feature_selection_mode": "手动选择特征",
        "features": [],
        "manual_confirm_trigger": 0,
        "current_tab": "手动筛选",
        "card_style": """
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        """,
        "page_flow": ["数据源", "变量定义与任务设置", "模型训练与评估"]
    }
    for k, v in init_states.items():
        st.session_state.setdefault(k, v)

    # -------------------------- 前置检查 --------------------------
    if not st.session_state.get("data_loaded", False) or st.session_state.df is None:
        st.warning("⚠️ 请先在【数据源】页面上传数据！")
        if st.button("← 返回数据源", use_container_width=True):
            st.session_state.current_page = st.session_state.page_flow[0]
            rerun()
        return

    df = st.session_state.df
    all_features = [c for c in df.columns if c not in {"Unnamed: 0", "index"}]
    st.session_state.all_features = all_features

    # -------------------------- 基础配置 --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("1. 基础配置")
        task_type = st.radio(
            "任务类型",
            ["分类任务", "回归任务"],
            index=0 if st.session_state.get("task_type") != "回归任务" else 1,
            help="分类=预测类别，回归=预测连续值"
        )
        st.session_state.task_type = task_type

        default_target_idx = 0
        if st.session_state.get("target_col") in all_features:
            default_target_idx = all_features.index(st.session_state.target_col)
        target_col = st.selectbox("目标变量（待预测列）", all_features, index=default_target_idx)
        st.session_state.target_col = target_col

        # 排除无关特征
        st.subheader("2. 排除无关特征（可选）")
        exclude_candidates = [f for f in all_features if f != target_col]
        if exclude_candidates:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("全选排除", use_container_width=True):
                    st.session_state.exclude_features = exclude_candidates.copy()
                    for lst in ["manual_features", "confirmed_manual_features", "auto_features",
                                "auto_features_executed"]:
                        st.session_state[lst] = [
                            f for f in st.session_state[lst]
                            if f not in exclude_candidates
                        ]
                    rerun()
            with c2:
                if st.button("取消所有排除", use_container_width=True):
                    st.session_state.exclude_features.clear()
                    rerun()

            cols = st.columns(min(3, len(exclude_candidates)))
            for idx, feat in enumerate(exclude_candidates):
                with cols[idx % 3]:
                    checked = feat in st.session_state.exclude_features
                    if st.checkbox(feat, value=checked, key=f"exclude_{feat}"):
                        if feat not in st.session_state.exclude_features:
                            st.session_state.exclude_features.append(feat)
                            for lst in ["manual_features", "confirmed_manual_features", "auto_features",
                                        "auto_features_executed"]:
                                if feat in st.session_state[lst]:
                                    st.session_state[lst].remove(feat)
                    else:
                        if feat in st.session_state.exclude_features:
                            st.session_state.exclude_features.remove(feat)

        candidate_features = [
            f for f in all_features
            if f != target_col and f not in st.session_state.exclude_features
        ]
        st.session_state.candidate_features = candidate_features
        if not st.session_state.manual_features and candidate_features:
            st.session_state.manual_features = candidate_features.copy()
            st.session_state.confirmed_manual_features = candidate_features.copy()
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- 特征筛选 --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("3. 特征筛选（两种方式独立）")
        st.info(f"候选特征：{len(candidate_features)} 个（已排除目标+{len(st.session_state.exclude_features)}个无关特征）")

        tab1, tab2 = st.tabs(["手动筛选", "自动筛选"])

        # 手动筛选
        with tab1:
            st.success("当前激活：手动筛选特征")
            if candidate_features:
                st.session_state.manual_features = [
                    f for f in st.session_state.manual_features if f in candidate_features
                ]
                man_cols = st.columns(min(3, len(candidate_features)))
                for idx, feat in enumerate(candidate_features):
                    with man_cols[idx % 3]:
                        sel = feat in st.session_state.manual_features

                        def toggle(f=feat, s=sel):
                            if s:
                                st.session_state.manual_features.remove(f)
                            else:
                                st.session_state.manual_features.append(f)

                        st.checkbox(feat, value=sel, key=f"manual_{feat}", on_change=toggle)

            col_confirm, _ = st.columns([1, 2])
            with col_confirm:
                if st.button("✅ 确认手动选择", use_container_width=True, type="primary"):
                    valid = [f for f in st.session_state.manual_features if f in candidate_features]
                    if not valid:
                        st.error("请至少选择1个特征后再确认")
                    else:
                        st.session_state.confirmed_manual_features = valid.copy()
                        st.session_state.active_selection_mode = "manual"
                        st.session_state.feature_selection_mode = "手动选择特征"
                        rerun()

            valid_confirmed = [f for f in st.session_state.confirmed_manual_features if f in candidate_features]
            if valid_confirmed:
                st.info(f"当前已确认的手动特征：{len(valid_confirmed)} 个")
                with st.expander("点击查看特征列表"):
                    st.write(", ".join(valid_confirmed))

        # 自动筛选（增加特征重要性可视化）
        with tab2:
            st.success("当前激活：自动筛选特征")
            if not candidate_features:
                st.warning("无候选特征可筛选")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    k = st.slider("保留特征数量", 1, len(candidate_features),
                                  min(5, len(candidate_features)), key="auto_k")
                    st.session_state.auto_select_params["k"] = k
                with col2:
                    if task_type == "分类任务":
                        method = st.selectbox("筛选方法",
                                              ["互信息（mutual_info_classif）",
                                               "卡方检验（chi2）",
                                               "Lasso回归（L1）"],
                                              key="auto_method_clf")
                    else:
                        method = st.selectbox("筛选方法",
                                              ["互信息（mutual_info_regression）",
                                               "皮尔逊相关（f_regression）",
                                               "Lasso回归（L1）"],
                                              key="auto_method_reg")
                    st.session_state.auto_select_params["method"] = method

                lasso_alpha = 0.1
                if "Lasso" in method:
                    lasso_alpha = st.slider("Lasso正则化强度（alpha）", 0.001, 1.0, 0.1, 0.001, key="auto_lasso_alpha")
                    st.session_state.auto_select_params["lasso_alpha"] = lasso_alpha

                if st.button("执行自动筛选", use_container_width=True, type="primary"):
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

                            selected_features, scores = [], []
                            if "互信息" in method:
                                sel = SelectKBest(
                                    mutual_info_classif if task_type == "分类任务" else mutual_info_regression,
                                    k=k
                                ).fit(X_scaled, y)
                                mask = sel.get_support()
                                selected_features = [f for f, m in zip(candidate_features, mask) if m]
                                scores = sel.scores_
                            elif "卡方" in method:
                                X_pos = X_scaled - X_scaled.min() + 1e-6
                                sel = SelectKBest(chi2, k=k).fit(X_pos, y)
                                mask = sel.get_support()
                                selected_features = [f for f, m in zip(candidate_features, mask) if m]
                                scores = sel.scores_
                            elif "皮尔逊" in method:
                                sel = SelectKBest(f_regression, k=k).fit(X_scaled, y)
                                mask = sel.get_support()
                                selected_features = [f for f, m in zip(candidate_features, mask) if m]
                                scores = sel.scores_
                            elif "Lasso" in method:
                                model = Lasso(alpha=lasso_alpha)
                                model.fit(X_scaled, y)
                                coef = np.abs(model.coef_)
                                top_idx = np.argsort(coef)[-k:] if np.sum(coef > 0) >= k else np.where(coef > 1e-8)[0]
                                selected_features = [candidate_features[i] for i in top_idx]
                                scores = coef

                            # 保存结果和分数
                            st.session_state["auto_features_executed"] = selected_features.copy()
                            st.session_state["auto_feature_scores"] = scores.copy()
                            st.session_state["active_selection_mode"] = "auto"
                            st.session_state["feature_selection_mode"] = "自动筛选特征"
                            st.session_state["current_tab"] = "自动筛选"

                            # 显示特征重要性排名可视化
                            st.success(f"自动筛选完成！保留 {len(selected_features)} 个特征")

                            # 创建特征重要性数据框并排序
                            importance_df = pd.DataFrame({
                                "特征": candidate_features,
                                "重要性分数": scores
                            }).sort_values(by="重要性分数", ascending=False)

                            # 显示前10个特征的重要性表格
                            st.subheader("特征重要性排名（前10名）")
                            st.dataframe(importance_df.head(10), use_container_width=True)

                            # 绘制特征重要性条形图
                            plt.figure(figsize=(10, 6))
                            plt.style.use('seaborn-v0_8-muted')

                            # 选择前10个最重要的特征进行可视化
                            top_features = importance_df.head(10)
                            sns.barplot(
                                x="重要性分数",
                                y="特征",
                                data=top_features,
                                palette="viridis"
                            )
                            plt.title(f"特征重要性排名（方法：{method}）", fontsize=14)
                            plt.xlabel("重要性分数", fontsize=12)
                            plt.ylabel("特征名称", fontsize=12)

                            # 在Streamlit中显示图表
                            st.pyplot(plt.gcf())
                            plt.close()

                            rerun()

                        except Exception as e:
                            st.error(f"筛选失败：{e}")

            # 显示已保存的自动筛选结果和可视化
            if st.session_state.get("auto_features_executed"):
                st.info(f"自动筛选已完成：{len(st.session_state.auto_features_executed)} 个特征")

                # 显示特征列表
                with st.expander("点击查看特征列表"):
                    st.write(", ".join(st.session_state.auto_features_executed))

                # 显示保存的特征重要性可视化
                if len(st.session_state.auto_feature_scores) > 0:
                    with st.expander("查看特征重要性排名"):
                        # 重建特征重要性数据框
                        importance_df = pd.DataFrame({
                            "特征": candidate_features,
                            "重要性分数": st.session_state.auto_feature_scores
                        }).sort_values(by="重要性分数", ascending=False)

                        st.dataframe(importance_df.head(10), use_container_width=True)

                        # 绘制特征重要性条形图
                        plt.figure(figsize=(10, 6))
                        plt.style.use('seaborn-v0_8-muted')

                        top_features = importance_df.head(10)
                        sns.barplot(
                            x="重要性分数",
                            y="特征",
                            data=top_features,
                            palette="viridis"
                        )
                        plt.title(f"特征重要性排名（方法：{st.session_state.auto_select_params['method']}）", fontsize=14)
                        plt.xlabel("重要性分数", fontsize=12)
                        plt.ylabel("特征名称", fontsize=12)

                        st.pyplot(plt.gcf())
                        plt.close()

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- 训练参数 + 导航 --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("4. 训练参数设置")
        st.session_state.test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.1)
        st.session_state.random_state = st.number_input("随机种子", 0, 1000, 42)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("← 返回数据源", use_container_width=True):
            st.session_state.current_page = st.session_state.page_flow[0]
            rerun()

    with col2:
        # 导航按钮优先读取“已执行”结果
        if st.session_state.get("active_selection_mode") == "auto":
            final_features = st.session_state.get("auto_features_executed", [])
            mode_text = "自动筛选特征"
        else:
            final_features = [f for f in st.session_state.confirmed_manual_features if f in candidate_features]
            mode_text = "手动选择特征"

        valid = len(final_features) > 0

        if valid:
            st.info(f"即将使用：{mode_text}（{len(final_features)} 个特征）")
            with st.expander("点击查看特征列表"):
                st.write(", ".join(final_features))
        else:
            if st.session_state.get("active_selection_mode") == "auto":
                st.warning("请先在自动筛选标签中设置参数并点击【执行自动筛选】按钮")
            else:
                st.warning("请先在手动筛选标签中完成特征选择并点击【确认手动选择】按钮")

        if st.button("下一步：模型训练与评估 →", use_container_width=True, type="primary", disabled=not valid):
            st.session_state.features = final_features.copy()
            st.session_state.feature_selection_mode = mode_text
            st.session_state.current_page = st.session_state.page_flow[2]
            st.success(f"已确认使用 {mode_text}，共 {len(final_features)} 个特征")
            rerun()
