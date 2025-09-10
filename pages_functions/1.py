else:
# 终极兜底：直接调用模型的predict_proba（绕开PyCaret预测结果）
st.warning("⚠️ 未从PyCaret预测结果中找到概率列，尝试用模型直接计算...")
try:
    # 提取原始模型，确保输入数据是数值型
    actual_model = extract_actual_model(model)

    # -------------------------- 核心增强：模型概率支持判断 --------------------------
    # 1. 检查模型是否支持predict_proba
    if hasattr(actual_model, 'predict_proba'):
        # 处理X_test：排除非数值列（避免模型报错）
        X_test_numeric = X_test.select_dtypes(include=[np.number]).fillna(0)
        # 调用predict_proba计算概率
        y_proba = actual_model.predict_proba(X_test_numeric)
        # 二分类取正类概率，多分类取概率矩阵
        if n_classes == 2:
            y_proba = y_proba[:, 1]  # 取正类（第二个维度）概率
        st.success(
            f"✅ 模型直接计算概率成功！{'二分类正类概率' if n_classes == 2 else f'多分类{len(classes)}个类别概率'}")
        y_proba = np.clip(y_proba, 0.0, 1.0)  # 修正概率范围（确保0-1之间）

    # 2. 若模型不支持predict_proba，给出具体解决方案
    else:
        # 判断模型类型，给出针对性提示
        model_type = str(type(actual_model)).lower()
        if "svc" in model_type:  # SVM模型
            st.error(
                "❌ 当前模型为SVM，默认不支持概率计算！请在模型训练时：\n1. 选择「选择单个模型」→ 「支持向量机」\n2. 确保训练代码中启用了probability=True")
        elif "kneighbors" in model_type:  # KNN模型
            st.error(
                "❌ 当前模型为KNN，默认不支持概率计算！请在模型训练时：\n1. 选择「选择单个模型」→ 「K近邻」\n2. 确保训练代码中启用了probability=True")
        else:  # 其他不支持概率的模型
            st.error(
                f"❌ 当前模型（{actual_model.__class__.__name__}）不支持概率计算，无法生成ROC曲线。建议更换为随机森林、XGBoost、LightGBM等支持概率的模型。")

except Exception as e:
    st.error(f"❌ 模型直接计算概率失败: {str(e)}")
    st.exception(e)  # 显示详细错误栈，便于定位问题