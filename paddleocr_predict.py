from paddleocr import PaddleOCR
from pprint import pprint
import cv2 as cv
import os
import streamlit as st                 # 用来快速做网页 UI（上传文件、显示结果）

@st.cache_resource
def init_ocr():
    """
    初始化PaddleOCR
    :return: 初始化之后的OCR模型
    """
    return PaddleOCR(use_angle_cls=True, lang='ch')

class paddleorc_predict():
    """
    用ocr提取车牌号字段并返回
    """
    def __init__(self, crop_result_dir, ocr_output_dir):
        self.crop_result_dir = crop_result_dir
        self.ocr_output_dir = ocr_output_dir



    def predict(self):
        """
        paddleocr模型提取车牌号并返回，存储paddleocr对输入图片的预处理结果
        :return: ocr模型提取之后的车牌号
        """
        folder_path = self.ocr_output_dir
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(folder_path, "已建立")

        ocr = init_ocr()
        results = ocr.predict(self.crop_result_dir)
        car_license_list = []
        # 将每个经过处理之后的车牌号图片保存,并且存储提取的车牌号字段
        img_name = 0
        for res in results:
            cv.imwrite(
                f"{self.ocr_output_dir}/{img_name}.jpg",
                res['doc_preprocessor_res']['output_img'],
            )
            car_license_list.append(res['rec_texts'])
            img_name += 1

        return car_license_list

if __name__ == '__main__':
    crop_path = "./out_results/crop_results"
    ocr_output_dir = "./out_results/ocr_results"
    ocr = paddleorc_predict(crop_result_dir=crop_path, ocr_output_dir=ocr_output_dir)
    print(ocr.predict())