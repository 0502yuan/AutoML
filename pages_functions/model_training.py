import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import (
    setup as setup_clf,
    create_model as create_model_clf,
    compare_models as compare_models_clf,
    pull as pull_clf,
    finalize_model as finalize_model_clf,
    save_model as save_model_clf,
    evaluate_model,
    predict_model as predict_model_clf,
    get_config as get_config_clf,
    tune_model as tune_model_clf
)
from pycaret.regression import (
    setup as setup_reg,
    create_model as create_model_reg,
    compare_models as compare_models_reg,
    pull as pull_reg,
    finalize_model as finalize_model_reg,
    save_model as save_model_reg,
    predict_model as predict_model_reg,
    get_config as get_config_reg,
    tune_model as tune_model_reg
)
from imblearn.over_sampling import SMOTE
import time
import os


def get_model_name(model):
    """提取模型名称"""
    return str(model).split('(')[0]


def get_actual_model(model):
    """获取实际模型对象"""
    if hasattr(model, 'estimator'):
        return model.estimator
    return model


def check_and_fix_class_imbalance(df, target_col, st_obj):
    """检查并处理类别不平衡问题"""
    try:
        class_dist = df[target_col].value_counts(normalize=True)
        min_class = class_dist.idxmin()
        min_ratio = class_dist.min()

        st_obj.write("类别分布情况:")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x=target_col, data=df, ax=ax)
        ax.set_title("目标变量类别分布")
        st_obj.pyplot(fig)

        if min_ratio < 0.1:
            st_obj.warning(f"检测到类别不平衡，最小类别占比仅为 {min_ratio:.2%}")
            handle_method = st_obj.radio(
                "选择处理方式:",
                ["不处理（可能影响模型性能）", "过采样（增加少数类样本）", "欠采样（减少多数类样本）"]
            )

            if handle_method == "过采样（增加少数类样本）":
                X = df.drop(columns=[target_col])
                y = df[target_col]
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
                st_obj.success("已使用SMOTE算法进行过采样处理")

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(x=target_col, data=df_resampled, ax=ax)
                ax.set_title("处理后的类别分布")
                st_obj.pyplot(fig)
                return df_resampled

            elif handle_method == "欠采样（减少多数类样本）":
                max_class = class_dist.idxmax()
                max_count = df[target_col].value_counts()[max_class]
                min_count = df[target_col].value_counts()[min_class]

                df_majority = df[df[target_col] == max_class].sample(n=min_count, random_state=42)
                df_minority = df[df[target_col] == min_class]
                df_resampled = pd.concat([df_majority, df_minority], axis=0).sample(frac=1, random_state=42)
                st_obj.success("已对多数类进行欠采样处理")

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(x=target_col, data=df_resampled, ax=ax)
                ax.set_title("处理后的类别分布")
                st_obj.pyplot(fig)
                return df_resampled

        return df

    except Exception as e:
        st_obj.error(f"处理类别不平衡时出错: {str(e)}")
        return None


def model_training_page():
    """模型训练与配置页面（稳定显示筛选结果）"""
    st.header("模型训练与配置")
    st.write("选择模型类型和训练参数，系统将自动完成模型训练与优化。")

    # -------------------------- 1. 初始化必要状态（避免KeyError） --------------------------
    init_states = {
        "model_training_mode": "选择单个模型",
        "selected_model": None,
        "model_short_name": None,
        "n_top_models": 3,
        "model_trained": False,
        "shap_success": False,
        "model_results": None
    }
    for key, val in init_states.items():
        if key not in st.session_state:
            st.session_state[key] = val

    current_mode = st.session_state.model_training_mode

    # -------------------------- 2. 前置检查（修复DataFrame判断歧义） --------------------------
    required_states = ['data_loaded', 'df', 'features', 'target_col', 'task_type', 'card_style',
                       'feature_selection_mode']
    missing_info = []
    for state in required_states:
        value = st.session_state.get(state)

        # 单独处理DataFrame（避免歧义）
        if state == 'df':
            if value is None or not isinstance(value, pd.DataFrame) or value.empty:
                missing_info.append("有效数据集（df为空或未加载）")

        # 处理特征列表
        elif state == 'features':
            if not isinstance(value, list) or len(value) == 0:
                missing_info.append("有效特征列表（未从变量定义页面获取）")

        # 处理筛选方式
        elif state == 'feature_selection_mode':
            if value not in ["手动选择特征", "自动筛选特征"]:
                missing_info.append("特征筛选方式（未正确设置）")

        # 处理其他状态
        else:
            if value is None or (isinstance(value, str) and len(value.strip()) == 0):
                missing_info.append(state)

    if missing_info:
        st.warning(f"⚠️ 请先完成【变量定义与任务设置】页面配置，缺少以下信息：{', '.join(missing_info)}")
        st.markdown("---")
        page_flow = st.session_state.page_flow
        current_idx = page_flow.index(st.session_state.current_page)
        if st.button("← 上一步：变量定义与任务设置", use_container_width=True):
            st.session_state.current_page = page_flow[current_idx - 1]
            st.rerun()
        return

    # -------------------------- 3. 读取变量定义页的最终结果 --------------------------
    selected_features = st.session_state.features  # 最终同步的特征
    feature_selection_mode = st.session_state.feature_selection_mode  # 最终同步的筛选方式
    target_col = st.session_state.target_col
    task_type = st.session_state.task_type
    auto_params = st.session_state.get("auto_select_params", {})
    df = st.session_state.df
    df_train = df[selected_features + [target_col]].copy()  # 构建训练数据集

    # -------------------------- 4. 显示上一步筛选信息（准确无误） --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("上一步特征筛选信息（来自变量定义页面）")

        # 显示筛选方式
        st.write("#### 1. 筛选方式")
        if feature_selection_mode == "自动筛选特征":
            st.success(
                f"✅ 自动筛选特征\n"
                f"• 筛选方法：{auto_params.get('method', '统计相关性分析')}\n"
                f"• 保留特征数：{auto_params.get('k', len(selected_features))}\n"
                f"• Lasso alpha：{auto_params.get('lasso_alpha', '不适用')}"
            )
        else:
            st.success(f"✅ 手动选择特征\n• 选择特征数：{len(selected_features)}")

        # 显示特征列表
        st.write("#### 2. 筛选后的特征列表")
        st.write(f"**特征总数**：{len(selected_features)} 个")
        cols = st.columns(3)
        for idx, feat in enumerate(selected_features):
            cols[idx % 3].write(f"• {feat}")

        # 显示任务信息
        st.write("#### 3. 目标变量与任务类型")
        st.write(f"**预测目标**：{target_col}\n**任务类型**：{task_type}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    # -------------------------- 5. 模型配置区域（无闪烁） --------------------------
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("模型配置")

        col1, col2 = st.columns(2)
        with col1:
            # 训练模式切换（无rerun，状态自动更新）
            new_mode = st.radio(
                "选择模型训练策略:",
                ["选择单个模型", "自动训练所有模型并选择最佳"],
                index=0 if current_mode == "选择单个模型" else 1,
                key="model_training_mode_radio"
            )

            # 仅在模式变化时更新状态（无频繁rerun）
            if new_mode != current_mode:
                st.session_state.model_training_mode = new_mode
                current_mode = new_mode
                if current_mode == "自动训练所有模型并选择最佳":
                    st.session_state.selected_model = None
                    st.session_state.model_short_name = None

            # 单个模型选择（无rerun）
            if current_mode == "选择单个模型":
                st.subheader("选择模型")
                model_options = {
                    "分类任务": {
                        "随机森林": "rf", "LightGBM": "lightgbm", "XGBoost": "xgboost",
                        "逻辑回归": "lr", "支持向量机": "svm", "梯度提升树": "gbc",
                        "决策树": "dt", "极端随机树": "et"
                    },
                    "回归任务": {
                        "随机森林": "rf", "LightGBM": "lightgbm", "XGBoost": "xgboost",
                        "线性回归": "lr", "支持向量机": "svm", "梯度提升树": "gbr",
                        "决策树": "dt", "极端随机树": "et"
                    }
                }[task_type]

                # 保留历史选择（无rerun）
                default_model_idx = 0
                if st.session_state.selected_model in model_options:
                    default_model_idx = list(model_options.keys()).index(st.session_state.selected_model)

                selected_model = st.selectbox(
                    "模型类型:",
                    options=list(model_options.keys()),
                    index=default_model_idx,
                    key="model_selector"
                )
                st.session_state.selected_model = selected_model
                st.session_state.model_short_name = model_options[selected_model]
            else:
                st.info("ℹ️ 系统将自动训练所有可用模型，超参调优后选择最佳模型")

        with col2:
            st.subheader("训练参数")
            # 复选框/滑块无rerun，状态自动保存
            use_tuning = st.checkbox("启用超参数优化", value=True)
            fix_imbalance = st.checkbox("自动处理类别不平衡", value=True)
            train_size = st.slider("训练集比例", 0.6, 0.9, 0.8, 0.05)

            # 顶级模型数量（仅自动模式显示，无rerun）
            if current_mode == "自动训练所有模型并选择最佳":
                n_top_models = st.slider(
                    "评估的顶级模型数量",
                    1, 5, st.session_state.n_top_models
                )
                st.session_state.n_top_models = n_top_models

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------- 6. 模型训练逻辑 --------------------------
    if st.button("开始训练模型", use_container_width=True, key="train_button"):
        # 处理类别不平衡
        if task_type == "分类任务" and fix_imbalance:
            df_balanced = check_and_fix_class_imbalance(df_train, target_col, st)
            if df_balanced is not None:
                df_train = df_balanced
            else:
                st.warning("⚠️ 类别不平衡处理失败，使用原始数据训练")

        try:
            with st.spinner("⏳ 模型训练中，请稍候..."):
                if task_type == "分类任务":
                    # 初始化分类环境
                    setup_config = setup_clf(
                        data=df_train,
                        target=target_col,
                        session_id=42,
                        train_size=train_size,
                        normalize=True,
                        numeric_imputation="median",
                        categorical_imputation="mode",
                        verbose=False,
                        html=False
                    )

                    # 训练逻辑
                    if current_mode == "选择单个模型":
                        model = create_model_clf(st.session_state.model_short_name, verbose=False)
                        st.info(f"✅ 已创建 {st.session_state.selected_model} 模型")
                    else:
                        with st.expander("所有模型初始性能比较", expanded=True):
                            st.subheader("所有模型初始性能（按F1排序）")
                            top_models = compare_models_clf(
                                n_select=st.session_state.n_top_models,
                                sort="F1",
                                verbose=False
                            )
                            # 统一包装成 list
                            top_models = [top_models] if st.session_state.n_top_models == 1 else top_models
                            model = top_models[0]
                        st.success(f"✅ 初始评估完成，最佳模型：{get_model_name(model)}")

                    # 超参调优
                    if use_tuning:
                        with st.expander("超参数优化过程", expanded=True):
                            st.subheader("模型超参优化（目标：F1）")
                            if current_mode == "自动训练所有模型并选择最佳":
                                tuned_models = []
                                for m in top_models:
                                    m_name = get_model_name(m)
                                    st.info(f"正在优化 {m_name}...")
                                    tuned_m = tune_model_clf(m, optimize="F1", n_iter=10, verbose=False)
                                    tuned_models.append(tuned_m)
                                    st.subheader(f"{m_name} 调优结果")
                                    st.dataframe(pull_clf(), use_container_width=True)
                                model = tuned_models[0]
                                st.success(f"✅ 调优完成，最佳模型：{get_model_name(model)}")
                            else:
                                model = tune_model_clf(model, optimize="F1", n_iter=10, verbose=False)
                                st.dataframe(pull_clf(), use_container_width=True)

                    # 保存结果
                    final_model = finalize_model_clf(model)
                    st.session_state.model_results = {
                        "model": final_model,
                        "X_train": get_config_clf("X_train"),
                        "X_test": get_config_clf("X_test"),
                        "y_train": get_config_clf("y_train"),
                        "y_test": get_config_clf("y_test"),
                        "pred_results": predict_model_clf(final_model, data=get_config_clf("X_test"))
                    }

                else:  # 回归任务
                    # 初始化回归环境
                    setup_config = setup_reg(
                        data=df_train,
                        target=target_col,
                        session_id=42,
                        train_size=train_size,
                        normalize=True,
                        numeric_imputation="median",
                        categorical_imputation="mode",
                        verbose=False,
                        html=False
                    )

                    # 训练逻辑
                    if current_mode == "选择单个模型":
                        model = create_model_reg(st.session_state.model_short_name, verbose=False)
                        st.info(f"✅ 已创建 {st.session_state.selected_model} 模型")
                    else:
                        with st.expander("所有模型初始性能比较", expanded=True):
                            st.subheader("所有模型初始性能（按RMSE排序）")
                            top_models = compare_models_reg(
                                n_select=st.session_state.n_top_models,
                                sort="RMSE",
                                verbose=False
                            )
                            st.dataframe(pull_reg(), use_container_width=True)
                        model = top_models[0]
                        st.success(f"✅ 初始评估完成，最佳模型：{get_model_name(model)}")

                    # 超参调优
                    if use_tuning:
                        with st.expander("超参数优化过程", expanded=True):
                            st.subheader("模型超参优化（目标：RMSE）")
                            if current_mode == "自动训练所有模型并选择最佳":
                                tuned_models = []
                                for m in top_models:
                                    m_name = get_model_name(m)
                                    st.info(f"正在优化 {m_name}...")
                                    tuned_m = tune_model_reg(m, optimize="RMSE", n_iter=10, verbose=False)
                                    tuned_models.append(tuned_m)
                                    st.subheader(f"{m_name} 调优结果")
                                    st.dataframe(pull_reg(), use_container_width=True)
                                model = tuned_models[0]
                                st.success(f"✅ 调优完成，最佳模型：{get_model_name(model)}")
                            else:
                                model = tune_model_reg(model, optimize="RMSE", n_iter=10, verbose=False)
                                st.dataframe(pull_reg(), use_container_width=True)

                    # 保存结果
                    final_model = finalize_model_reg(model)
                    st.session_state.model_results = {
                        "model": final_model,
                        "X_train": get_config_reg("X_train"),
                        "X_test": get_config_reg("X_test"),
                        "y_train": get_config_reg("y_train"),
                        "y_test": get_config_reg("y_test"),
                        "pred_results": predict_model_reg(final_model, data=get_config_reg("X_test"))
                    }

                # 标记训练完成
                st.session_state.model = st.session_state.model_results["model"]
                st.session_state.model_trained = True
                st.success("✅ 模型训练完成！请前往【模型评估与解释】页面查看结果")

        except Exception as e:
            st.error(f"❌ 模型训练错误: {str(e)}")
            st.exception(e)
            st.session_state.model_trained = False
            if "model_results" in st.session_state:
                del st.session_state["model_results"]

    # -------------------------- 7. 导航按钮 --------------------------
    st.markdown("---")
    page_flow = st.session_state.page_flow
    current_idx = page_flow.index(st.session_state.current_page)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 上一步：变量定义与任务设置", use_container_width=True):
            st.session_state.current_page = page_flow[current_idx - 1]
            st.rerun()

    with col2:
        if st.session_state.get("model_trained", False):
            if st.button("下一步：模型评估与解释 →", use_container_width=True, type="primary"):
                st.session_state.current_page = page_flow[current_idx + 1]
                st.rerun()
        else:
            st.button("下一步：模型评估与解释 →", use_container_width=True, type="primary", disabled=True,
                      help="请先完成模型训练")