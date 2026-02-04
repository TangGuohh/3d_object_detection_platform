import streamlit as st
import pandas as pd
import os
import tempfile
from modules.api_client import APIClient
from modules.image_utils import ImageUtils
from modules.visualization import Visualization3D
from modules.visualization_2d import Visualization2D
from modules.camera_utils import CameraUtils


# 页面配置
st.set_page_config(
    page_title="3D目标检测平台",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化工具类
api_client = APIClient()
image_utils = ImageUtils()
visualizer_3d = Visualization3D()
visualizer_2d = Visualization2D()
camera_utils = CameraUtils()

# 初始化session state
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['图片名称', '理解结果', '检测模式'])
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_image_name' not in st.session_state:
    st.session_state.current_image_name = None

# 标题
st.title("📦 目标检测平台")

# 侧边栏
with st.sidebar:
    st.header("配置")

    # 图片上传
    uploaded_file = st.file_uploader(
        "上传图片", 
        type=['png', 'jpg', 'jpeg'],
        help="支持 PNG, JPG, JPEG 格式"
    )

    # 检测模式选择
    detection_mode = st.radio(
        "检测模式",
        ["图片理解", "2D目标检测", "3D目标检测"],
        help="选择不同的检测模式"
    )

    # 提示词输入
    prompt = st.text_area(
        "提示词",
        placeholder="请输入对图片的提问或要检测的物体名称...",
        help="在目标检测模式下，请输入要检测的物体名称"
    )

    # 目标检测专用设置
    if detection_mode in ["2D目标检测", "3D目标检测"]:
        st.info("💡 目标检测模式下，请在提示词中输入要检测的物体名称")

        # 检测类别设置
        if not prompt:
            st.warning("请输入要检测的物体名称")

    # API选择
    api_type = st.selectbox(
        "选择API服务",
        ["modelscope", "dashscope"],
        help="选择使用的AI服务提供商"
    )

    # 高分辨率选项
    high_resolution = st.checkbox("高分辨率模式", value=False)

    # 相机参数设置（仅3D模式）
    if detection_mode == "3D目标检测":
        st.subheader("相机参数设置")
        fov = st.slider("视场角 (FOV)", 30, 120, 60)

    # 处理按钮
    process_button = st.button("开始处理", type="primary")

# 处理上传的图片
if uploaded_file is not None:
    # 保存上传的图片
    image_path, image_name = image_utils.save_uploaded_file(uploaded_file)
    st.session_state.current_image = image_path
    st.session_state.current_image_name = image_name

    # 显示上传的图片
    resized_image = image_utils.resize_image(image_path)
    st.sidebar.image(resized_image, caption=f"上传的图片: {image_name}", use_column_width=True)

# 处理按钮点击事件
if process_button and st.session_state.current_image:
    if not prompt and detection_mode == "图片理解":
        st.warning("请输入提示词")
    else:
        with st.spinner("正在处理图片..."):
            try:
                # 构建最终的提示词
                if detection_mode == "图片理解":
                    final_prompt = prompt
                elif detection_mode == "2D目标检测":
                    final_prompt = f"找出图片中的所有{prompt}。针对每个{prompt}，提供其2D边界框，要求以JSON格式输出: [{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"类别名称\"}}]"
                else:  # 3D目标检测
                    final_prompt = f"找出图片中的所有{prompt}。针对每个{prompt}，提供其3D边界框，要求以JSON格式输出: [{{\"bbox_3d\":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],\"label\":\"category\"}}]"

                # 调用API
                response = api_client.inference_with_api(
                    image_path=st.session_state.current_image,
                    prompt=final_prompt,
                    api_type=api_type,
                    high_resolution=high_resolution
                )

                st.success("处理完成！")

                # 主界面布局
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("上传的图片")
                    resized_image = image_utils.resize_image(st.session_state.current_image)
                    st.image(resized_image, use_column_width=True)

                with col2:
                #    st.subheader("分析结果")

                    if detection_mode == "图片理解":
                        st.write(response)
                    elif detection_mode == "2D目标检测":

                        # 显示原始响应（用于调试）
                        #with st.expander("查看原始响应"):
                        #    st.text(response)

                        # 解析2D边界框
                        parsed_response = api_client.parse_json_response(response)
                        bbox_2d_results = api_client.parse_bbox_2d_from_text(response)

                        st.subheader("解析后的2D边界框")
                        #st.json(bbox_2d_results)

                        # 可视化2D边界框
                        with st.spinner("生成2D可视化结果..."):
                            fig = visualizer_2d.draw_bboxes_2d(
                                st.session_state.current_image, 
                                bbox_2d_results
                            )
                            st.pyplot(fig)

                    else:  # 3D目标检测
                        # 显示原始响应（用于调试）
                        #with st.expander("查看原始响应"):
                        #    st.text(response)

                        # 解析3D边界框
                        parsed_response = api_client.parse_json_response(response)
                        bbox_3d_results = api_client.parse_bbox_3d_from_text(response)

                        st.subheader("解析后的3D边界框")
                        #st.json(bbox_3d_results)

                        # 可视化3D边界框
                        with st.spinner("生成3D可视化结果..."):
                            # 获取相机参数
                            if st.session_state.current_image_name and camera_utils.load_camera_params(st.session_state.current_image_name):
                                cam_params = camera_utils.load_camera_params(st.session_state.current_image_name)
                            else:
                                cam_params = camera_utils.generate_camera_params(
                                    st.session_state.current_image, 
                                    fov=fov
                                )

                            # 绘制3D边界框
                            fig = visualizer_3d.draw_3dbboxes(
                                st.session_state.current_image, 
                                cam_params, 
                                bbox_3d_results
                            )

                            if fig is not None:
                                st.pyplot(fig)
                            else:
                                st.warning("3D可视化生成失败")

                # 添加到历史记录
                new_record = pd.DataFrame({
                    '图片名称': [st.session_state.current_image_name],
                    '理解结果': [response],
                    '检测模式': [detection_mode]
                })
                st.session_state.history = pd.concat([st.session_state.history, new_record], ignore_index=True)

            except Exception as e:
                st.error(f"处理过程中出现错误: {str(e)}")

# 显示历史记录
st.markdown("---")
st.subheader("历史记录")

if not st.session_state.history.empty:
    # 显示历史记录表格
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

    # 清空历史记录按钮
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("清空历史记录"):
            st.session_state.history = pd.DataFrame(columns=['图片名称', '理解结果', '检测模式'])
            st.rerun()
    with col2:
        # 导出历史记录
        csv = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="导出历史记录为CSV",
            data=csv,
            file_name="detection_history.csv",
            mime="text/csv"
        )
else:
    st.info("暂无历史记录")

# 使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 🎯 平台使用指南

    **1. 上传图片**
    - 支持 PNG、JPG、JPEG 格式
    - 图片大小建议不超过10MB

    **2. 选择检测模式**
    - **图片理解**: 普通的图片内容理解和分析
    - **2D目标检测**: 检测并定位图片中的物体（2D边界框）
    - **3D目标检测**: 检测并估计物体的3D空间位置

    **3. 输入提示词**
    - 图片理解模式: 输入任意图片理解问题
    - 目标检测模式: 输入要检测的物体名称

    **4. 高级设置**
    - 选择API服务提供商
    - 高分辨率模式（处理细节丰富的图像）
    - 3D模式下的相机参数调整

    ### 🔧 技术特点

    - **多模态AI**: 基于Qwen3-VL强大的视觉语言理解能力
    - **3D空间感知**: 精确估计物体在3D空间中的位置和姿态
    - **实时交互**: 即时显示检测结果和可视化效果
    - **历史追溯**: 完整记录每次检测的分析结果

    ### 💡 应用场景

    - 自动驾驶场景的车辆和行人检测
    - 室内场景的物体定位和空间理解
    - 机器人视觉导航
    - AR/VR应用开发
    - 智能监控和分析
    """)