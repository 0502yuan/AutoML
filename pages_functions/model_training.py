import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
)
# 分类模型
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# 回归模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def rerun():
    """兼容不同Streamlit版本的刷新方法"""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()


def model_training_page(df):
    """模型训练与配置页面（接收df参数）"""
    st.header("模型训练与配置")

    # 初始化会话状态
    init_states = {
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "best_model": None,
        "model_performance": {},
        "test_size": 0.2,
        "random_state": 42,
        "card_style": """
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        """
    }
    for k, v in init_states.items():
        st.session_state.setdefault(k, v)

    # 检查必要的会话状态变量
    required_states = ["target_col", "features", "task_type"]
    for state in required_states:
        if state not in st.session_state or st.session_state[state] is None:
            st.error(f"❌ 缺少必要的配置：{state}，请返回上一步完成设置")
            if st.button("← 返回变量定义页面", use_container_width=True):
                st.session_state.current_page = "变量定义与任务设置"
                rerun()
            return

    # 提取特征和目标变量
    try:
        X = df[st.session_state.features].copy()
        y = df[st.session_state.target_col].copy()
        st.success(f"✅ 数据准备完成：{X.shape[0]}个样本，{X.shape[1]}个特征")
    except KeyError as e:
        st.error(f"❌ 特征列不存在：{str(e)}")
        if st.button("← 返回变量定义页面", use_container_width=True):
            st.session_state.current_page = "变量定义与任务设置"
            rerun()
        return

    # 显示数据集信息
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("1. 数据集信息")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**任务类型**：", st.session_state.task_type)
            st.write("**目标变量**：", st.session_state.target_col)
            st.write("**特征数量**：", len(st.session_state.features))

        with col2:
            st.write("**样本数量**：", X.shape[0])
            st.write("**特征列表**：",
                     ", ".join(st.session_state.features[:5]) + ("..." if len(st.session_state.features) > 5 else ""))

        # 数据集拆分配置
        st.subheader("2. 数据集拆分")
        test_size = st.slider(
            "测试集比例",
            min_value=0.1,
            max_value=0.4,
            value=st.session_state.test_size,
            step=0.05,
            help="训练集与测试集的划分比例"
        )
        st.session_state.test_size = test_size

        # 执行数据集拆分
        if st.button("🔄 拆分训练集与测试集", use_container_width=True):
            with st.spinner("正在拆分数据集..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=st.session_state.random_state,
                    stratify=y if st.session_state.task_type == "分类任务" else None
                )

                # 保存到会话状态
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                st.success(
                    f"✅ 数据集拆分完成：\n"
                    f"训练集：{X_train.shape[0]}个样本，{X_train.shape[1]}个特征\n"
                    f"测试集：{X_test.shape[0]}个样本，{X_test.shape[1]}个特征"
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # 模型选择与训练
    if st.session_state.X_train is not None:
        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("3. 模型选择与训练")

            # 根据任务类型选择模型
            if st.session_state.task_type == "分类任务":
                model_name = st.selectbox(
                    "选择分类模型",
                    ["逻辑回归", "决策树", "随机森林", "梯度提升树"]
                )

                # 模型参数
                with st.expander("模型参数设置", expanded=False):
                    if model_name == "逻辑回归":
                        C = st.slider("正则化强度 (C)", 0.01, 10.0, 1.0, 0.01)
                        max_iter = st.slider("最大迭代次数", 100, 1000, 500, 100)
                        model = LogisticRegression(C=C, max_iter=max_iter, random_state=st.session_state.random_state)

                    elif model_name == "决策树":
                        max_depth = st.slider("树最大深度", 3, 20, 5)
                        min_samples_split = st.slider("最小分裂样本数", 2, 20, 2)
                        model = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=st.session_state.random_state
                        )

                    elif model_name == "随机森林":
                        n_estimators = st.slider("树的数量", 50, 500, 100, 50)
                        max_depth = st.slider("树最大深度", 3, 20, 5)
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=st.session_state.random_state
                        )

                    elif model_name == "梯度提升树":
                        n_estimators = st.slider("树的数量", 50, 500, 100, 50)
                        learning_rate = st.slider("学习率", 0.01, 0.3, 0.1, 0.01)
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            random_state=st.session_state.random_state
                        )

            else:  # 回归任务
                model_name = st.selectbox(
                    "选择回归模型",
                    ["线性回归", "Ridge回归", "Lasso回归", "决策树", "随机森林", "梯度提升树"]
                )

                # 模型参数
                with st.expander("模型参数设置", expanded=False):
                    if model_name == "线性回归":
                        model = LinearRegression()

                    elif model_name == "Ridge回归":
                        alpha = st.slider("正则化强度 (alpha)", 0.01, 10.0, 1.0, 0.01)
                        model = Ridge(alpha=alpha, random_state=st.session_state.random_state)

                    elif model_name == "Lasso回归":
                        alpha = st.slider("正则化强度 (alpha)", 0.01, 10.0, 1.0, 0.01)
                        model = Lasso(alpha=alpha, random_state=st.session_state.random_state)

                    elif model_name == "决策树":
                        max_depth = st.slider("树最大深度", 3, 20, 5)
                        min_samples_split = st.slider("最小分裂样本数", 2, 20, 2)
                        model = DecisionTreeRegressor(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=st.session_state.random_state
                        )

                    elif model_name == "随机森林":
                        n_estimators = st.slider("树的数量", 50, 500, 100, 50)
                        max_depth = st.slider("树最大深度", 3, 20, 5)
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=st.session_state.random_state
                        )

                    elif model_name == "梯度提升树":
                        n_estimators = st.slider("树的数量", 50, 500, 100, 50)
                        learning_rate = st.slider("学习率", 0.01, 0.3, 0.1, 0.01)
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            random_state=st.session_state.random_state
                        )

            # 训练模型
            if st.button("🚀 训练模型", use_container_width=True, type="primary"):
                with st.spinner(f"正在训练{model_name}..."):
                    # 特征预处理
                    numeric_features = st.session_state.X_train.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_features = st.session_state.X_train.select_dtypes(include=['object']).columns.tolist()

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_features),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                        ])

                    # 创建Pipeline
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])

                    # 训练模型
                    pipeline.fit(st.session_state.X_train, st.session_state.y_train)

                    # 保存模型
                    st.session_state.best_model = pipeline

                    # 评估模型
                    y_pred = pipeline.predict(st.session_state.X_test)

                    # 计算性能指标
                    performance = {}
                    if st.session_state.task_type == "分类任务":
                        performance["准确率"] = accuracy_score(st.session_state.y_test, y_pred)
                        performance["精确率"] = precision_score(st.session_state.y_test, y_pred, average='weighted')
                        performance["召回率"] = recall_score(st.session_state.y_test, y_pred, average='weighted')
                        performance["F1分数"] = f1_score(st.session_state.y_test, y_pred, average='weighted')

                        # 尝试计算AUC（多类别的情况下可能不适用）
                        try:
                            if len(np.unique(st.session_state.y_test)) <= 2:  # 二分类
                                y_pred_proba = pipeline.predict_proba(st.session_state.X_test)[:, 1]
                                performance["AUC"] = roc_auc_score(st.session_state.y_test, y_pred_proba)
                        except:
                            pass
                    else:  # 回归任务
                        performance["MSE"] = mean_squared_error(st.session_state.y_test, y_pred)
                        performance["RMSE"] = np.sqrt(performance["MSE"])
                        performance["MAE"] = mean_absolute_error(st.session_state.y_test, y_pred)
                        performance["R²分数"] = r2_score(st.session_state.y_test, y_pred)

                    st.session_state.model_performance = performance
                    st.success(f"✅ {model_name}训练完成！")
            st.markdown("</div>", unsafe_allow_html=True)

    # 模型评估结果
    if st.session_state.best_model is not None and st.session_state.model_performance:
        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("4. 模型评估结果")

            # 显示性能指标
            col1, col2 = st.columns(2)
            metrics = list(st.session_state.model_performance.items())
            for i, (name, value) in enumerate(metrics):
                if i % 2 == 0:
                    with col1:
                        st.metric(name, f"{value:.4f}")
                else:
                    with col2:
                        st.metric(name, f"{value:.4f}")

            # 可视化结果
            st.subheader("5. 结果可视化")
            if st.session_state.task_type == "分类任务":
                # 混淆矩阵
                y_pred = st.session_state.best_model.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=np.unique(st.session_state.y_test),
                            yticklabels=np.unique(st.session_state.y_test))
                plt.xlabel('预测标签')
                plt.ylabel('真实标签')
                plt.title('混淆矩阵')
                st.pyplot(plt)

                # 分类报告
                with st.expander("查看详细分类报告", expanded=False):
                    report = classification_report(
                        st.session_state.y_test,
                        y_pred,
                        target_names=[str(c) for c in np.unique(st.session_state.y_test)]
                    )
                    st.text(report)
            else:
                # 回归结果可视化：预测值 vs 真实值
                y_pred = st.session_state.best_model.predict(st.session_state.X_test)

                plt.figure(figsize=(8, 6))
                plt.scatter(st.session_state.y_test, y_pred, alpha=0.6)
                plt.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                         [st.session_state.y_test.min(), st.session_state.y_test.max()],
                         'r--')
                plt.xlabel('真实值')
                plt.ylabel('预测值')
                plt.title('预测值 vs 真实值')
                st.pyplot(plt)

                # 残差图
                plt.figure(figsize=(8, 6))
                residuals = st.session_state.y_test - y_pred
                plt.scatter(y_pred, residuals, alpha=0.6)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('预测值')
                plt.ylabel('残差')
                plt.title('残差图')
                st.pyplot(plt)
            st.markdown("</div>", unsafe_allow_html=True)

    # 导航按钮
    with st.container():
        page_flow = st.session_state.page_flow
        try:
            current_idx = page_flow.index(st.session_state.current_page)
        except ValueError:
            current_idx = 2  # 模型训练页面默认索引

        col_prev, col_next = st.columns(2)
        with col_prev:
            if current_idx > 0:
                if st.button(f"← 上一步：{page_flow[current_idx - 1]}", use_container_width=True):
                    st.session_state.current_page = page_flow[current_idx - 1]
                    rerun()
