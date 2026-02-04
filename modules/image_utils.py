import os
import uuid
from PIL import Image
import io

class ImageUtils:
    @staticmethod
    def save_uploaded_file(uploaded_file, upload_dir="uploaded_images"):
        """
        保存上传的文件到指定目录

        参数:
            uploaded_file: 上传的文件对象，需要包含name属性和getbuffer方法
            upload_dir: 保存文件的目录路径，默认为"uploaded_images"

        返回:
            tuple: 包含文件完整路径和唯一文件名的元组 (file_path, unique_filename)
        """
        # 检查并创建上传目录
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # 生成唯一文件名
        file_extension = os.path.splitext(uploaded_file.name)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path, unique_filename

    @staticmethod
    def resize_image(image_path, max_size=(800, 600)):
        """
        调整图片大小以适应显示

        参数:
            image_path: 图片文件的路径
            max_size: 图片最大尺寸元组(width, height)，默认为(800, 600)

        返回:
            PIL.Image: 调整大小后的Image对象
        """
        # 打开图片并调整大小
        image = Image.open(image_path)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # 处理透明背景图片
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background

        return image