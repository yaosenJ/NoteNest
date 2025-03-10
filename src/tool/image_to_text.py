"""
pip install pytesseract pdf2image

# ����������б�
sudo apt-get update

# ��װ poppler-utils������ pdfinfo �ȹ��ߣ�
sudo apt-get install -y poppler-utils

# ��װ Tesseract OCR ����
sudo apt-get install -y tesseract-ocr

# ��װ�������԰�
sudo apt-get install -y tesseract-ocr-chi-sim


"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter


# ��� Linux �����µ� Tesseract ·������
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


def preprocess_image(image):
    # ת��Ϊ�Ҷ�ͼ��
    image = image.convert('L')
    # ��ǿ�Աȶ�
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Ӧ���˾�
    image = image.filter(ImageFilter.MedianFilter())
    return image

def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(
        preprocessed_image,
        lang='chi_sim',       # ����ʶ��
        config='--oem 3 --psm 6'  # �Ż�ʶ��ģʽ
    )
    return text

# ָ����Ҫ�����ҳ����Χ
start_page = 3
end_page = 4
pdf_path = './����.pdf'
# ����ָ��ҳ����PDFҳ��
images = convert_from_path(pdf_path, poppler_path="/usr/bin" )
selected_images = images[start_page - 1:end_page]

# ��ȡ�ı�
ocr_texts = [extract_text_from_image(image) for image in selected_images]

# �����ȡ���ı�
for page_num, text in enumerate(ocr_texts, start=start_page):
    print(f"Page {page_num}:\n{text}\n")
