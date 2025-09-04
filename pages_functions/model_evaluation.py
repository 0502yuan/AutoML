# model_evaluation_page.py  2025-09-04  修复版
# 修复：1) 蜂群图维度错误  2) ROC 曲线异常（二分类/多分类兼容）

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


# ------------------------- 工具函数 -------------------------
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
    required = ['model_trained', 'model_results', 'task_type', 'target_col', 'model']
    missing = [r for r in required if r not in st.session_state or st.session_state[r] is None]
    if missing:
        st.warning(f"⚠️ 请先完成模型训练，确保: {', '.join(missing)}")
        page_flow = st.session_state.page_flow
        cur_idx = page_flow.index(st.session_state.current_page)
        if st.button("← 返回模型训练", use_container_width=True):
            st.session_state.current_page = page_flow[cur_idx - 1]
            st.rerun()
        return

    # 2. 提取数据
    results = st.session_state.model_results
    model = st.session_state.model
    actual_model = extract_actual_model(model)
    X_test = results["X_test"]
    y_test = results["y_test"]
    pred_results = results["pred_results"]
    THRESHOLD = 0.5

    model_name = str(actual_model.__class__.__name__)
    st.success(f"当前评估模型: {model_name}")

    # 3. 提取预测结果
    y_pred = None
    y_proba = None
    classes = sorted(y_test.unique())
    n_classes = len(classes)

    if st.session_state.task_type == "分类任务":
        # 预测标签
        pred_label_cols = [c for c in pred_results.columns if 'prediction_label' in c]
        if not pred_label_cols:
            pred_label_cols = [c for c in pred_results.columns if c != st.session_state.target_col]
        if pred_label_cols:
            y_pred = pred_results[pred_label_cols[0]]

            # 预测概率
            pred_prob_cols = [c for c in pred_results.columns if 'prediction_score' in c]
            if pred_prob_cols:
                if n_classes == 2:
                    y_proba = pred_results[pred_prob_cols[-1]].values  # 正类概率
                else:
                    y_proba = pred_results[pred_prob_cols].values
            else:
                st.warning("未检测到预测概率列，AUC 等指标无法计算")
        else:
            st.error("未找到预测标签列")
            return

    # 4. 模型信息
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("1. 模型信息与参数")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**最优模型**")
            st.write(f"模型类型: {model_name}")
            st.write(f"任务类型: {st.session_state.task_type}")
            st.write(f"分类阈值: {THRESHOLD}")
        with col2:
            params = get_model_parameters(actual_model)
            params_df = pd.DataFrame(list(params.items())[:10], columns=["参数名", "值"])
            st.dataframe(params_df, use_container_width=True)
            if len(params) > 10:
                st.caption(f"显示前10个参数，共{len(params)}个")
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. 性能指标
    if st.session_state.task_type == "分类任务" and y_pred is not None:
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
                if n_classes == 2:
                    pos_precision = report.get('1', report[str(classes[-1])])['precision']
                    pos_recall = report.get('1', report[str(classes[-1])])['recall']
                    st.metric("正类精确率", f"{pos_precision:.4f}")
                    st.metric("正类召回率", f"{pos_recall:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    # 6. 可视化
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("3. 模型评估可视化")
        if st.session_state.task_type == "分类任务" and y_pred is not None:
            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel('预测标签')
            ax.set_ylabel('实际标签')
            ax.set_title('混淆矩阵')
            st.pyplot(fig)

            # ROC 曲线
            if y_proba is not None:
                if n_classes == 2:
                    # 二分类
                    if y_proba.ndim > 1:
                        y_proba_pos = y_proba[:, 1]
                    else:
                        y_proba_pos = y_proba
                    fpr, tpr, _ = roc_curve(y_test, y_proba_pos, pos_label=classes[1])
                    roc_auc_val = auc(fpr, tpr)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_val:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('假正率')
                    ax.set_ylabel('真正率')
                    ax.set_title('ROC 曲线')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    # 多分类 One-vs-Rest
                    from sklearn.preprocessing import label_binarize
                    y_bin = label_binarize(y_test, classes=classes)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                        auc_val = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{cls} (AUC={auc_val:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('假正率')
                    ax.set_ylabel('真正率')
                    ax.set_title('多分类 ROC (One-vs-Rest)')
                    ax.legend()
                    st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # 7. SHAP 解释
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

                    # 维度统一：多分类取正类 / 降维
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]  # 正类
                    if shap_vals.ndim == 3:
                        shap_vals = shap_vals[:, :, 1]  # 降三维 → 二维

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

    # 8. 导航
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