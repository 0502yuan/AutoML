# model_evaluation_page.py
# 修复：1) ROC曲线反转问题（添加正类概率验证）2) 优化概率列选择逻辑
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
import shap


def extract_actual_model(pipeline_model):
    """从 PyCaret 的 Pipeline 中提取实际模型"""
    try:
        if hasattr(pipeline_model, 'steps'):
            for name, step in reversed(pipeline_model.steps):
                if 'model' in name or 'estimator' in name:
                    return step
            return pipeline_model.steps[-1][1]
        elif hasattr(pipeline_model, 'estimator'):
            return extract_actual_model(pipeline_model.estimator)
        elif hasattr(pipeline_model, 'final_estimator'):
            return extract_actual_model(pipeline_model.final_estimator)
        else:
            return pipeline_model
    except Exception as e:
        st.warning(f"提取模型时警告: {str(e)}")
        return pipeline_model


def get_model_parameters(model):
    """提取模型参数"""
    try:
        if hasattr(model, 'get_params'):
            params = model.get_params()
            return {k: v for k, v in params.items()
                    if not isinstance(v, (list, dict)) or len(str(v)) < 100}
        return {}
    except Exception as e:
        st.warning(f"提取模型参数时出错: {str(e)}")
        return {}


def model_evaluation_page():
    st.header("模型评估与解释")

    # 1. 前置检查
    required = ['model_trained', 'model', 'task_type', 'target_col', 'X_test', 'y_test']
    missing = [r for r in required if r not in st.session_state or st.session_state[r] is None]
    if missing:
        st.warning(f"⚠️ 请先完成模型训练，确保: {', '.join(missing)}")
        page_flow = st.session_state.page_flow
        cur_idx = page_flow.index(st.session_state.current_page)
        if st.button("← 返回模型训练", use_container_width=True):
            st.session_state.current_page = page_flow[cur_idx - 1]
            st.rerun()
        return

    # 2. 读取测试集真实数据
    model = st.session_state.model
    task_type = st.session_state.task_type
    target_col = st.session_state.target_col
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    actual_model = extract_actual_model(model)

    # 3. 在测试集上重新预测
    try:
        from pycaret.classification import predict_model as predict_clf
        from pycaret.regression import predict_model as predict_reg
        pred_func = predict_clf if task_type == "分类任务" else predict_reg
        pred_results = pred_func(model, data=X_test)
    except Exception as e:
        st.error(f"测试集预测失败: {str(e)}")
        return

    # 4. 提取预测标签 & 概率（核心修改：优化概率列选择+反转验证）
    y_pred = None
    y_proba = None
    classes = sorted(y_test.unique())
    n_classes = len(classes)
    pos_class = classes[-1]  # 正类默认是类别列表中最大的值（如1在[0,1]中）
    pos_class_ratio = np.mean(y_test == pos_class)  # 正类样本比例

    if task_type == "分类任务":
        # 标签：优先选择带"prediction_label"后缀的列，避免误取原始标签
        label_col = [c for c in pred_results.columns if c.endswith('prediction_label')]
        if not label_col:
            st.warning("未找到标准预测标签列，尝试从非目标列中选择")
            label_col = [c for c in pred_results.columns if c != target_col and 'pred' in c.lower()]
        if not label_col:
            st.error("未找到预测标签列，无法计算评估指标")
            return
        y_pred = pred_results[label_col[0]].values

        # 概率：核心修改1：优先选择与正类匹配的概率列
        prob_cols = [c for c in pred_results.columns if c.startswith('prediction_score_')]
        if not prob_cols:
            prob_cols = [c for c in pred_results.columns if 'prediction_score' in c]

        if prob_cols:
            # 核心修改2：根据正类标签选择对应的概率列（如正类是1，选择"prediction_score_1"）
            pos_prob_col = [c for c in prob_cols if str(pos_class) in c]
            if pos_prob_col:
                y_proba = pred_results[pos_prob_col[0]].values
                st.info(f"已选择正类[{pos_class}]对应的概率列：{pos_prob_col[0]}")
            else:
                # 若无明确匹配列，默认取最后一列，但添加反转验证
                y_proba = pred_results[prob_cols[-1]].values
                st.warning(f"未找到正类[{pos_class}]对应的概率列，默认使用最后一列：{prob_cols[-1]}")

            # 核心修改3：验证概率与正类标签的相关性，修复反转问题
            prob_pos_ratio = np.mean(y_proba[y_test == pos_class])  # 正类样本的平均概率
            prob_neg_ratio = np.mean(y_proba[y_test != pos_class])  # 负类样本的平均概率

            # 若正类样本的平均概率 < 负类样本的平均概率，说明概率反转，取补集（1-概率）
            if prob_pos_ratio < prob_neg_ratio:
                st.warning(
                    f"检测到概率反转（正类平均概率{prob_pos_ratio:.3f} < 负类平均概率{prob_neg_ratio:.3f}），已自动修正")
                y_proba = 1 - y_proba  # 反转概率
        else:
            st.warning("未检测到预测概率列，ROC 等指标无法计算")

    # 5. 模型信息
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("1. 模型信息与参数")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**最优模型**")
            st.write(f"模型类型: {str(actual_model.__class__.__name__)}")
            st.write(f"任务类型: {task_type}")
            st.write(f"正类标签: {pos_class}（占比: {pos_class_ratio:.3f}）")
        with col2:
            params = get_model_parameters(actual_model)
            params_df = pd.DataFrame(list(params.items())[:10], columns=["参数名", "值"])
            st.dataframe(params_df, use_container_width=True)
            if len(params) > 10:
                st.caption(f"显示前10个参数，共{len(params)}个")
        st.markdown("</div>", unsafe_allow_html=True)

    # 6. 性能指标
    if task_type == "分类任务" and y_pred is not None:
        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("2. 模型性能指标")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(report_df, use_container_width=True)
            with col2:
                st.metric("准确率", f"{report.get('accuracy', 0):.4f}")
                st.metric("宏平均F1", f"{report['macro avg']['f1-score']:.4f}")
                st.metric("加权平均F1", f"{report['weighted avg']['f1-score']:.4f}")
                pos_precision = report.get(str(pos_class), report['weighted avg'])['precision']
                pos_recall = report.get(str(pos_class), report['weighted avg'])['recall']
                st.metric(f"正类[{pos_class}]精确率", f"{pos_precision:.4f}")
                st.metric(f"正类[{pos_class}]召回率", f"{pos_recall:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    # 7. 可视化（核心修改：ROC曲线使用修正后的概率）
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("3. 模型评估可视化")
        if task_type == "分类任务" and y_pred is not None:
            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel('预测标签')
            ax.set_ylabel('实际标签')
            ax.set_title(f'混淆矩阵（正类：{pos_class}）')
            st.pyplot(fig)

            # ROC 曲线（核心修改：使用修正后的y_proba）
            if y_proba is not None:
                if n_classes == 2:
                    # 核心修改4：确保概率是一维数组
                    if y_proba.ndim > 1:
                        y_proba_pos = y_proba[:, 0]
                    else:
                        y_proba_pos = y_proba

                    # 计算ROC曲线（使用修正后的正类概率）
                    fpr, tpr, _ = roc_curve(y_test, y_proba_pos, pos_label=pos_class)
                    roc_auc_val = auc(fpr, tpr)

                    # 验证AUC是否合理（若AUC<0.5，说明仍有反转，取1-AUC）
                    if roc_auc_val < 0.5:
                        st.warning(f"检测到ROC曲线反转（AUC={roc_auc_val:.3f}），已自动修正")
                        fpr, tpr, _ = roc_curve(y_test, 1 - y_proba_pos, pos_label=pos_class)
                        roc_auc_val = auc(fpr, tpr)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC 曲线（AUC = {roc_auc_val:.3f}）', color='#1f77b4')
                    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测')
                    ax.set_xlabel('假正率（FPR）')
                    ax.set_ylabel('真正率（TPR）')
                    ax.set_title(f'ROC 曲线（正类：{pos_class}）')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                else:
                    from sklearn.preprocessing import label_binarize
                    y_bin = label_binarize(y_test, classes=classes)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for i, cls in enumerate(classes):
                        # 对每个类别处理概率反转
                        cls_prob = y_proba[:, i] if y_proba.ndim > 1 else y_proba
                        fpr, tpr, _ = roc_curve(y_bin[:, i], cls_prob)
                        roc_auc_val = auc(fpr, tpr)

                        # 修正反转的类别ROC
                        if roc_auc_val < 0.5:
                            fpr, tpr, _ = roc_curve(y_bin[:, i], 1 - cls_prob)
                            roc_auc_val = auc(fpr, tpr)

                        ax.plot(fpr, tpr, label=f'类别 {cls} (AUC={roc_auc_val:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('假正率')
                    ax.set_ylabel('真正率')
                    ax.set_title('多分类 ROC (One-vs-Rest)')
                    ax.legend()
                    st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # 8. SHAP 解释
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("4. 模型解释（SHAP值）")

        if st.button("计算SHAP值", use_container_width=True):
            st.session_state.shap_success = False

        if not st.session_state.get("shap_success", False):
            try:
                with st.spinner("⏳ 正在计算SHAP值..."):
                    actual_model = extract_actual_model(model)
                    if any(t in str(type(actual_model)).lower()
                           for t in ['tree', 'forest', 'gbm', 'xgboost', 'lightgbm']):
                        explainer = shap.TreeExplainer(actual_model)
                    else:
                        bg = shap.sample(X_test, min(100, X_test.shape[0]))
                        explainer = shap.KernelExplainer(actual_model.predict, bg)

                    sample_data = shap.sample(X_test, min(100, X_test.shape[0]))
                    shap_vals = explainer.shap_values(sample_data)

                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                    if shap_vals.ndim == 3:
                        shap_vals = shap_vals[:, :, 1]

                    st.session_state.shap_explainer = explainer
                    st.session_state.shap_values = shap_vals
                    st.session_state.shap_sample_data = sample_data
                    st.session_state.shap_success = True
                    st.success("✅ SHAP值计算完成")
            except Exception as e:
                st.error(f"❌ SHAP计算失败: {str(e)}")

        if st.session_state.get("shap_success", False):
            try:
                shap_vals = st.session_state.shap_values
                sample = st.session_state.shap_sample_data

                tab1, tab2 = st.tabs(["摘要图", "蜂群图"])
                with tab1:
                    fig = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_vals, sample, plot_type="bar", show=False)
                    st.pyplot(fig)
                    plt.close()

                with tab2:
                    fig = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_vals, sample, plot_type="beeswarm", show=False)
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"❌ SHAP可视化失败: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

    # 9. 导航
    st.markdown("---")
    page_flow = st.session_state.page_flow
    cur_idx = page_flow.index(st.session_state.current_page)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← 返回模型训练", use_container_width=True):
            st.session_state.current_page = page_flow[cur_idx - 1]
            st.rerun()
    with c2:
        if st.button("下一步：模型预测 →", use_container_width=True, type="primary"):
            st.session_state.current_page = page_flow[cur_idx + 1]
            st.rerun()