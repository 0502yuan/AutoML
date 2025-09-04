# data_processing.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def check_and_fix_class_imbalance(df, target_col, st):
    """检查目标变量类别分布，处理样本数过少的类别"""
    with st.container():
        st.markdown(f"<div style='{st.session_state.card_style}'>", unsafe_allow_html=True)
        st.subheader("目标变量类别分布")

        class_counts = df[target_col].value_counts()
        class_df = class_counts.reset_index(name="样本数").rename(columns={target_col: "类别"})

        # 可视化类别分布
        fig, ax = plt.subplots()
        sns.barplot(x="类别", y="样本数", data=class_df, ax=ax)
        ax.set_title("类别分布条形图")
        st.pyplot(fig)

        st.dataframe(class_df, use_container_width=True)

        min_samples = class_counts.min()
        if min_samples < 2:
            st.warning(f"⚠️ 发现极端类别不平衡：最少的类别仅包含 {min_samples} 个样本")

            if len(class_counts) == 2:
                st.error("❌ 二分类任务中存在类别样本数为1的情况，无法直接训练，请补充数据或合并类别！")
                st.markdown("</div>", unsafe_allow_html=True)
                return None
            else:
                smallest_class = class_counts.idxmin()
                st.info(f"🔧 自动处理：将最小类别 '{smallest_class}' 合并到样本数最多的类别中")
                largest_class = class_counts.idxmax()
                df[target_col] = df[target_col].replace(smallest_class, largest_class)
                st.success(f"✅ 处理后类别分布：\n{df[target_col].value_counts()}")
                st.markdown("</div>", unsafe_allow_html=True)
                return df
        else:
            st.success(f"✅ 类别分布正常，最小类别样本数：{min_samples}")
            st.markdown("</div>", unsafe_allow_html=True)
            return df
