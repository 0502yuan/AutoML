# utils.py
import shap
import numpy as np
from sklearn.pipeline import Pipeline


def get_actual_model(pipeline_model):
    """从PyCaret的Pipeline中提取实际模型"""
    if isinstance(pipeline_model, Pipeline):
        # 遍历Pipeline步骤，找到最后一个模型组件
        for name, step in reversed(pipeline_model.steps):
            if 'classifier' in name or 'regressor' in name or 'model' in name:
                return step
        return pipeline_model.steps[-1][1]
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


def get_explainer_and_shap_values(model, actual_model, X_test_transformed, task_type, st):
    """获取合适的SHAP解释器和SHAP值，增加数据验证"""
    # 明确支持TreeExplainer的模型
    tree_explainer_supported = [
        '随机森林', 'XGBoost', 'LightGBM',
        '梯度提升树', '决策树'
    ]

    model_name = get_model_name(model)

    # 数据验证：确保输入数据有效
    if X_test_transformed.empty:
        st.error("❌ 预处理后的测试数据为空，无法进行SHAP分析")
        return None, None, None, None

    # 确保数据是数值型
    if not np.issubdtype(X_test_transformed.dtypes.iloc[0], np.number):
        st.error("❌ 预处理后的测试数据包含非数值类型，无法进行SHAP分析")
        return None, None, None, None

    try:
        # 尝试使用TreeExplainer（速度快）
        if model_name in tree_explainer_supported:
            explainer = shap.TreeExplainer(actual_model)
            # 处理分类任务的SHAP值（多分类时返回列表）
            if task_type == "分类":
                # 二分类取第一个类别，多分类取所有
                shap_values = explainer.shap_values(X_test_transformed)
                # 二分类时取正类的SHAP值
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # 取正类的SHAP值
            else:
                shap_values = explainer.shap_values(X_test_transformed)
            return explainer, shap_values, "TreeExplainer", X_test_transformed

        # 对于其他模型，使用KernelExplainer
        else:
            st.info(f"ℹ️ 模型 {model_name} 不支持TreeExplainer，使用KernelExplainer（速度较慢）")

            # 数据采样（确保样本量有效）
            sample_size = min(100, len(X_test_transformed))
            if sample_size < 5:
                st.warning(f"⚠️ 样本量过少（仅{sample_size}个），可能影响SHAP分析效果")
            background = X_test_transformed.sample(sample_size, random_state=42)

            # 初始化解释器
            if task_type == "分类":
                explainer = shap.KernelExplainer(actual_model.predict_proba, background)
            else:
                explainer = shap.KernelExplainer(actual_model.predict, background)

            # 预测数据采样
            predict_sample_size = min(100, len(X_test_transformed))
            X_sample = X_test_transformed.sample(predict_sample_size, random_state=42)

            # 计算SHAP值
            shap_values = explainer.shap_values(X_sample)
            # 处理分类任务的SHAP值
            if task_type == "分类" and isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # 二分类取正类
            return explainer, shap_values, "KernelExplainer", X_sample

    except Exception as e:
        st.error(f"❌ SHAP解释器初始化失败: {str(e)}")
        return None, None, None, None
