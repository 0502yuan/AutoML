# pages/model_management.py
import streamlit as st
import os
import datetime
import joblib
from utils import get_model_name


def save_model_page():
    """模型保存页面（尾页：仅显示上一步）"""
    st.header("模型管理")
    st.write("保存当前训练的模型或管理已保存的模型。")

    if not st.session_state.model_trained and st.session_state.model is None:
        st.warning("⚠️ 没有可用的模型！请先训练模型")
    else:
        current_model_name = get_model_name(st.session_state.model) if st.session_state.model else "未知模型"

        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("保存当前模型")

            if st.session_state.model:
                st.info(f"当前模型: {current_model_name}")

            model_name_input = st.text_input("输入模型名称",
                                             f"{st.session_state.task_type}_model_{current_model_name}",
                                             key="model_name")

            model_desc = st.text_area("模型描述（可选）",
                                      f"{st.session_state.task_type}模型，目标变量：{st.session_state.target_col}，基础模型：{current_model_name}",
                                      key="model_desc")

            if st.button("保存模型", key="save_model_button", use_container_width=True):
                try:
                    if not os.path.exists("saved_models"):
                        os.makedirs("saved_models")

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{model_name_input}_{timestamp}.pkl"
                    filepath = os.path.join("saved_models", filename)

                    joblib.dump(st.session_state.model, filepath)

                    meta_data = {
                        "name": model_name_input,
                        "base_model": current_model_name,
                        "task_type": st.session_state.task_type,
                        "target_col": st.session_state.target_col,
                        "features": st.session_state.features,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "description": model_desc
                    }
                    meta_path = os.path.join("saved_models", f"{model_name_input}_{timestamp}_meta.pkl")
                    joblib.dump(meta_data, meta_path)

                    st.success(f"✅ 模型已保存为: {filename}")

                    if filename not in st.session_state.saved_models:
                        st.session_state.saved_models.append(filename)

                except Exception as e:
                    st.error(f"❌ 模型保存失败: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
            st.subheader("📂 已保存的模型")

            if not os.path.exists("saved_models"):
                os.makedirs("saved_models")
            saved_models = [f for f in os.listdir("saved_models") if f.endswith(".pkl") and not f.endswith("_meta.pkl")]
            st.session_state.saved_models = saved_models

            if saved_models:
                model_list = []
                for model_file in saved_models:
                    try:
                        meta_file = model_file.replace(".pkl", "_meta.pkl")
                        meta_path = os.path.join("saved_models", meta_file)
                        if os.path.exists(meta_path):
                            meta_data = joblib.load(meta_path)
                            model_list.append({
                                "模型名称": meta_data.get("name", model_file),
                                "基础模型": meta_data.get("base_model", "未知"),
                                "任务类型": meta_data.get("task_type", "未知"),
                                "目标变量": meta_data.get("target_col", "未知"),
                                "保存时间": meta_data.get("timestamp", "未知"),
                                "文件名": model_file
                            })
                        else:
                            model_list.append({
                                "模型名称": model_file,
                                "基础模型": "未知",
                                "任务类型": "未知",
                                "目标变量": "未知",
                                "保存时间": "未知",
                                "文件名": model_file
                            })
                    except:
                        model_list.append({
                            "模型名称": model_file,
                            "基础模型": "未知",
                            "任务类型": "未知",
                            "目标变量": "未知",
                            "保存时间": "未知",
                            "文件名": model_file
                        })

                model_df = model_list
                st.dataframe([{k: v for k, v in m.items() if k != "文件名"} for m in model_df],
                             use_container_width=True)

                st.subheader("模型操作")
                selected_model = st.selectbox("选择模型进行操作:", saved_models)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("删除选中模型", key="delete_model_button"):
                        try:
                            model_path = os.path.join("saved_models", selected_model)
                            os.remove(model_path)

                            meta_file = selected_model.replace(".pkl", "_meta.pkl")
                            meta_path = os.path.join("saved_models", meta_file)
                            if os.path.exists(meta_path):
                                os.remove(meta_path)

                            st.success(f"✅ 模型 '{selected_model}' 已删除")
                            # 刷新页面以更新模型列表
                            st.session_state.current_page = st.session_state.current_page

                        except Exception as e:
                            st.error(f"❌ 删除失败: {e}")

                with col2:
                    if st.button("下载选中模型", key="download_model_button"):
                        try:
                            model_path = os.path.join("saved_models", selected_model)
                            with open(model_path, "rb") as file:
                                st.download_button(
                                    label="下载模型文件",
                                    data=file,
                                    file_name=selected_model,
                                    mime="application/octet-stream",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"❌ 下载失败: {e}")
            else:
                st.info("ℹ️ 尚未保存任何模型，训练模型后可在此处保存")
            st.markdown("</div>", unsafe_allow_html=True)

    # 上一步按钮（尾页无"下一步"）
    st.markdown("---")
    page_flow = st.session_state.page_flow
    current_idx = page_flow.index(st.session_state.current_page)

    # 布局：左（上一步）、右（空）
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 上一步：模型预测", use_container_width=True):
            st.session_state.current_page = page_flow[current_idx - 1]
            st.rerun()
