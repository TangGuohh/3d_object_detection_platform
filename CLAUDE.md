# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit-based 3D object detection platform that uses Qwen3-VL vision-language models (via ModelScope or DashScope) to detect and visualize objects in images. Supports three modes: general image understanding, 2D object detection, and 3D object detection with spatial awareness.

## Running the Application

```bash
pip install streamlit openai pillow numpy opencv-python matplotlib python-dotenv
streamlit run app.py
```

Test script (for API testing):
```bash
python test.py
```

## Architecture

**Main Application (`app.py`):**
- Streamlit UI with session state for history tracking
- Three detection modes: 图片理解, 2D目标检测, 3D目标检测
- Sidebar for configuration: image upload, mode selection, prompt input, API type selection, FOV slider (for 3D mode)
- History logging with CSV export

**Modules:**

| Module | Purpose |
|--------|---------|
| `api_client.py` | API communication with ModelScope/DashScope; handles Base64 encoding, JSON parsing, bbox extraction |
| `visualization.py` | 3D to 2D projection using camera intrinsics and Euler angle rotation (yaw, pitch, roll); draws 3D boxes on 2D images |
| `visualization_2d.py` | Draws 2D bounding boxes with color-coded labels; handles multiple bbox formats (dict, list, JSON string) |
| `camera_utils.py` | Loads camera params from `spatial_understanding/cam_infos.json` or generates them from FOV and image dimensions |
| `image_utils.py` | File upload handling with UUID naming; image resizing with thumbnail generation; transparent background handling |

## API Response Formats

**2D Detection Prompt:**
```
找出图片中的所有{object_name}。针对每个{object_name}，提供其2D边界框，要求以JSON格式输出: [{"bbox_2d": [x1, y1, x2, y2], "label": "类别名称"}]
```

**3D Detection Prompt:**
```
找出图片中的所有{object_name}。针对每个{object_name}，提供其3D边界框，要求以JSON格式输出: [{"bbox_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw], "label": "category"}]
```

## Important Notes

- **Security**: API keys are currently hardcoded in `api_client.py:21` - should use `.env` file instead
- **3D Visualization**: Requires camera parameters. Falls back to generated params if no `cam_infos.json` exists
- **Chinese Font**: All visualizations configured with matplotlib fallback fonts (SimHei, Microsoft YaHei)
- **Image Storage**: Uploaded images saved to `uploaded_images/` directory
- **ModelScope Default**: `Qwen/Qwen3-VL-235B-A22B-Instruct`
- **DashScope Default**: `qwen3-vl-plus` (requires `DASHSCOPE_API_KEY` env var)
