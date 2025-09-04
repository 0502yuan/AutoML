# pages_functions/model_prediction.py
import streamlit as st
import pandas as pd
import os
import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import predict_model as predict_model_clf
from pycaret.regression import predict_model as predict_model_reg
from utils import get_actual_model


def model_prediction_page():
    """模型预测页面"""
    st.header("模型预测")
    st.write("使用训练好的模型对新数据进行预测，支持多种输入方式。")

    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("模型选择")
        model_option = st.radio("选择预测模型:",
                                ["当前训练的模型", "已保存的模型"],
                                key="model_option")

        if model_option == "已保存的模型":
            if not os.path.exists("saved_models"):
                os.makedirs("saved_models")

            saved_models = [f for f in os.listdir("saved_models") if f.endswith(".pkl")]
            st.session_state.saved_models = saved_models

            if not saved_models:
                st.warning("⚠️ 没有找到已保存的模型！请先训练并保存模型")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                selected_model = st.selectbox("选择已保存的模型:", saved_models)

                if st.button("加载模型", key="load_model_button"):
                    try:
                        with st.spinner("⏳ 加载模型中..."):
                            model_path = os.path.join("saved_models", selected_model)
                            model = joblib.load(model_path)
                            actual_model = get_actual_model(model)
                            st.session_state.model = model
                            st.session_state.actual_model = actual_model
                            st.success(f"✅ 模型 '{selected_model}' 加载成功！")
                    except Exception as e:
                        st.error(f"❌ 模型加载失败: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ 没有可用的模型！请先训练模型或加载已保存的模型")
    else:
        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("预测数据输入")
            input_option = st.radio("选择输入方式:",
                                    ["使用当前数据集", "上传新数据", "手动输入"],
                                    key="input_option")

            predict_df = None
            if input_option == "使用当前数据集" and st.session_state.df is not None:
                predict_df = st.session_state.df.copy()
                st.success("✅ 已加载当前数据集作为预测数据")

            elif input_option == "上传新数据":
                new_file = st.file_uploader("上传预测数据 (CSV格式)", type="csv")
                if new_file:
                    with st.spinner("⏳ 加载数据中..."):
                        try:
                            # 尝试多种常见编码格式
                            encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'utf-16']
                            predict_df = None
                            used_encoding = None

                            for encoding in encodings_to_try:
                                try:
                                    # 重置文件指针到开头
                                    new_file.seek(0)
                                    predict_df = pd.read_csv(new_file, encoding=encoding)
                                    used_encoding = encoding
                                    break
                                except UnicodeDecodeError:
                                    continue

                            if predict_df is None:
                                st.error("❌ 无法解析文件，尝试了多种编码格式均失败，请检查文件是否为有效的CSV格式")
                            else:
                                st.success(f"✅ 预测数据加载成功，使用编码: {used_encoding}")
                        except Exception as e:
                            st.error(f"❌ 数据加载错误: {str(e)}")
                else:
                    st.warning("⚠️ 请上传预测数据！")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                num_rows = st.number_input("输入样本数量", min_value=1, max_value=20, value=3, key="num_rows")
                predict_df = pd.DataFrame(columns=st.session_state.features)

                for i in range(num_rows):
                    with st.expander(f"样本 #{i + 1} 输入", expanded=False):
                        row_data = {}
                        for feature in st.session_state.features:
                            try:
                                if pd.api.types.is_numeric_dtype(st.session_state.df[feature]):
                                    row_data[feature] = st.number_input(
                                        f"{feature}",
                                        key=f"{feature}_{i}"
                                    )
                                else:
                                    row_data[feature] = st.text_input(
                                        f"{feature}",
                                        key=f"{feature}_{i}"
                                    )
                            except:
                                row_data[feature] = st.text_input(
                                    f"{feature}",
                                    key=f"{feature}_{i}"
                                )
                        predict_df = pd.concat([predict_df, pd.DataFrame([row_data])], ignore_index=True)
                st.success("✅ 手动输入完成")

            if predict_df is not None and not predict_df.empty:
                st.subheader("预览预测数据")
                st.dataframe(predict_df.head(), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if st.button("执行预测", use_container_width=True, key="predict_button"):
                    with st.spinner("⏳ 正在执行预测..."):
                        try:
                            if st.session_state.task_type == "分类":
                                predictions = predict_model_clf(st.session_state.model, data=predict_df)
                            else:
                                predictions = predict_model_reg(st.session_state.model, data=predict_df)

                            with st.container():
                                st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
                                st.subheader("预测结果")

                                if st.session_state.task_type == "分类":
                                    result_df = predictions[["prediction_label", "prediction_score"]]
                                    st.dataframe(result_df, use_container_width=True)

                                    st.subheader("预测结果分布")
                                    fig, ax = plt.subplots()
                                    predictions["prediction_label"].value_counts().plot(kind='bar', ax=ax)
                                    ax.set_title("预测类别分布")
                                    ax.set_xlabel("类别")
                                    ax.set_ylabel("样本数")
                                    st.pyplot(fig)
                                else:
                                    result_df = predictions[["prediction_label"]].rename(
                                        columns={"prediction_label": "预测值"}
                                    )
                                    st.dataframe(result_df, use_container_width=True)

                                    st.subheader("预测值分布")
                                    fig, ax = plt.subplots()
                                    sns.histplot(predictions["prediction_label"], kde=True, ax=ax)
                                    ax.set_title("预测值分布")
                                    ax.set_xlabel("预测值")
                                    st.pyplot(fig)

                                csv = predictions.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="下载预测结果",
                                    data=csv,
                                    file_name=f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key="download_predictions",
                                    use_container_width=True
                                )
                                st.markdown("</div>", unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"❌ 预测失败: {e}")
            else:
                st.warning("⚠️ 没有可用于预测的数据！")
                st.markdown("</div>", unsafe_allow_html=True)

    # 上一步+下一步按钮
    st.markdown("---")
    page_flow = st.session_state.page_flow
    current_idx = page_flow.index(st.session_state.current_page)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 上一步：模型评估与解释", use_container_width=True):
            st.session_state.current_page = page_flow[current_idx - 1]
            st.rerun()

    with col2:
        if st.button("下一步：模型管理 →", use_container_width=True, type="primary"):
            st.session_state.current_page = page_flow[current_idx + 1]
            st.rerun()
