# utils.py
import shap
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


def get_actual_model(pipeline_model):
    """从PyCaret的Pipeline中提取实际模型（增强嵌套支持）"""
    if isinstance(pipeline_model, Pipeline):
        # 遍历Pipeline步骤，找到最后一个模型组件
        for name, step in reversed(pipeline_model.steps):
            # 处理嵌套Pipeline
            if isinstance(step, Pipeline):
                return get_actual_model(step)
            # 处理包含estimator的包装器（如SMOTE）
            elif hasattr(step, 'estimator') and isinstance(step.estimator, BaseEstimator):
                return get_actual_model(step.estimator)
            # 直接匹配模型关键词
            elif any(keyword in name for keyword in ['classifier', 'regressor', 'model']) and isinstance(step,
                                                                                                         BaseEstimator):
                return step
        # 若未找到明确模型，返回最后一步
        if pipeline_model.steps:
            return get_actual_model(pipeline_model.steps[-1][1])
        return pipeline_model
    # 处理单独的模型
    elif hasattr(pipeline_model, 'estimator') and isinstance(pipeline_model.estimator, BaseEstimator):
        return get_actual_model(pipeline_model.estimator)
    elif isinstance(pipeline_model, BaseEstimator):
        return pipeline_model
    return pipeline_model


def get_model_name(model):
    """获取模型的中文名称"""
    actual_model = get_actual_model(model)
    model_class_name = actual_model.__class__.__name__

    # 模型名称映射表
    name_mapping = {
        # 分类模型
        'RandomForestClassifier': '随机森林',
        'LGBMClassifier': 'LightGBM',
        'XGBClassifier': 'XGBoost',
        'LogisticRegression': '逻辑回归',
        'SVC': '支持向量机',
        'GradientBoostingClassifier': '梯度提升树',
        'DecisionTreeClassifier': '决策树',
        'ExtraTreesClassifier': '极端随机树',
        # 回归模型
        'RandomForestRegressor': '随机森林',
        'LGBMRegressor': 'LightGBM',
        'XGBRegressor': 'XGBoost',
        'LinearRegression': '线性回归',
        'SVR': '支持向量机',
        'GradientBoostingRegressor': '梯度提升树',
        'DecisionTreeRegressor': '决策树',
        'ExtraTreesRegressor': '极端随机树'
    }
    return name_mapping.get(model_class_name, model_class_name)


def get_explainer_and_shap_values(model, X_test_transformed, task_type, st):
    """获取合适的SHAP解释器和SHAP值（增强错误处理）"""
    # 明确支持TreeExplainer的模型
    tree_explainer_supported = [
        '随机森林', 'XGBoost', 'LightGBM',
        '梯度提升树', '决策树', '极端随机树'
    ]

    model_name = get_model_name(model)
    actual_model = get_actual_model(model)

    # 数据验证：确保输入数据有效
    if X_test_transformed is None or X_test_transformed.empty:
        st.error("❌ 预处理后的测试数据为空，无法进行SHAP分析")
        return None, None, None, None

    # 确保数据是数值型
    non_numeric_cols = X_test_transformed.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        st.warning(f"⚠️ 检测到非数值特征：{non_numeric_cols}，将自动删除")
        X_test_transformed = X_test_transformed.select_dtypes(include=[np.number])
        if X_test_transformed.empty:
            st.error("❌ 删除非数值特征后数据为空，无法进行SHAP分析")
            return None, None, None, None

    try:
        # 尝试使用TreeExplainer（速度快）
        if model_name in tree_explainer_supported:
            st.info(f"ℹ️ 使用TreeExplainer分析 {model_name} 模型")
            explainer = shap.TreeExplainer(actual_model)

            # 处理分类任务的SHAP值
            if task_type == "分类任务":
                shap_values = explainer.shap_values(X_test_transformed)
                # 二分类时取正类的SHAP值（PyCaret默认正类为1）
                if isinstance(shap_values, list):
                    # 检查是否为二分类（2个类别）
                    if len(shap_values) == 2:
                        shap_values = shap_values[1]  # 取正类
                    # 多分类时取第一个类别（可根据需求调整）
                    else:
                        shap_values = shap_values[0]
            else:
                shap_values = explainer.shap_values(X_test_transformed)

            # 验证SHAP值维度
            if shap_values.shape[0] != X_test_transformed.shape[0] or shap_values.shape[1] != X_test_transformed.shape[
                1]:
                st.warning("⚠️ SHAP值维度与输入数据不匹配，将重新采样")
                sample_size = min(50, X_test_transformed.shape[0])
                X_sample = X_test_transformed.sample(sample_size, random_state=42)
                shap_values = explainer.shap_values(X_sample)
                if task_type == "分类任务" and isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                return explainer, shap_values, "TreeExplainer", X_sample

            return explainer, shap_values, "TreeExplainer", X_test_transformed

        # 对于其他模型，使用KernelExplainer
        else:
            st.info(f"ℹ️ 模型 {model_name} 不支持TreeExplainer，使用KernelExplainer（速度较慢）")

            # 数据采样（确保样本量有效）
            sample_size = min(100, X_test_transformed.shape[0])
            if sample_size < 5:
                st.warning(f"⚠️ 样本量过少（仅{sample_size}个），将使用全部样本")
                sample_size = X_test_transformed.shape[0]
            background = X_test_transformed.sample(sample_size, random_state=42)

            # 初始化解释器（根据任务类型选择预测函数）
            st.info(f"ℹ️ 使用 {sample_size} 个样本作为背景数据")
            if task_type == "分类任务":
                # 确保模型有predict_proba方法
                if hasattr(actual_model, 'predict_proba'):
                    explainer = shap.KernelExplainer(
                        lambda x: actual_model.predict_proba(x)[:, 1],  # 二分类正类概率
                        background
                    )
                else:
                    st.error("❌ 分类模型缺少predict_proba方法，无法计算SHAP值")
                    return None, None, None, None
            else:
                explainer = shap.KernelExplainer(actual_model.predict, background)

            # 预测数据采样（避免计算量过大）
            predict_sample_size = min(50, X_test_transformed.shape[0])
            X_sample = X_test_transformed.sample(predict_sample_size, random_state=42)

            # 计算SHAP值
            st.spinner("⏳ 正在计算SHAP值（KernelExplainer速度较慢，请耐心等待）")
            shap_values = explainer.shap_values(X_sample)

            return explainer, shap_values, "KernelExplainer", X_sample

    except Exception as e:
        st.error(f"❌ SHAP解释器初始化失败: {str(e)}")
        st.exception(e)  # 显示详细错误信息
        return None, None, None, None