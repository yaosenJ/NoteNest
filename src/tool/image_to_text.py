"""
pip install pytesseract pdf2image

# 更新软件包列表
sudo apt-get update

# 安装 poppler-utils（包含 pdfinfo 等工具）
sudo apt-get install -y poppler-utils

# 安装 Tesseract OCR 引擎
sudo apt-get install -y tesseract-ocr

# 安装中文语言包
sudo apt-get install -y tesseract-ocr-chi-sim


"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter


# 添加 Linux 环境下的 Tesseract 路径配置
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


def preprocess_image(image):
    # 转换为灰度图像
    image = image.convert('L')
    # 增强对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # 应用滤镜
    image = image.filter(ImageFilter.MedianFilter())
    return image

def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(
        preprocessed_image,
        lang='chi_sim',       # 中文识别
        config='--oem 3 --psm 6'  # 优化识别模式
    )
    return text

# 指定需要处理的页数范围
start_page = 3
end_page = 4
pdf_path = './北京.pdf'
# 加载指定页数的PDF页面
images = convert_from_path(pdf_path, poppler_path="/usr/bin" )
selected_images = images[start_page - 1:end_page]

# 提取文本
ocr_texts = [extract_text_from_image(image) for image in selected_images]

# 输出提取的文本
for page_num, text in enumerate(ocr_texts, start=start_page):
    print(f"Page {page_num}:\n{text}\n")
