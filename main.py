from model_predict import model_predict
from paddleocr_predict import paddleorc_predict
import streamlit as st
import numpy as np
import cv2 as cv
import os

if __name__ == '__main__':
    # yolov11模型路径
    model_path = "runs/detect/train/weights/best.pt"
    # 图片中间结果输出文件夹路径
    out_dir = r".\out_results"
    # 上传的图片存储路径
    original_picture_path = r".\out_results\original"
    # 裁剪之后的图片存储路径
    crop_picture_path = r'.\out_results\crop_results'
    # ocr检测之后的图片存储路径
    ocr_picture_path = r'.\out_results\ocr_results'
    if not os.path.exists(original_picture_path):
        os.mkdir(original_picture_path)
        print(original_picture_path, "已建立")
    # 页面基础设置（标题、布局）
    st.set_page_config(
        page_title="车牌号提取 OCR DEMO",
        layout="wide",
        page_icon="car"
    )
    st.title("车牌号提取 OCR DEMO")

    # --------- Streamlit 文件上传控件 ----------
    uploaded_file = st.file_uploader("上传车辆图片(jpg)", type=["jpg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='上传的车辆图片')
        # 对上传的图片数据流进行转化显示
        car_img_bytes = uploaded_file.read()
        print(type(car_img_bytes))
        car_img_array = np.frombuffer(car_img_bytes, np.uint8)
        print(car_img_array.shape)
        car_img = cv.imdecode(car_img_array, cv.IMREAD_COLOR)
        print(car_img.shape)
        # 存储上传的原始图像
        cv.imwrite(f"{original_picture_path}/original.jpg", car_img)
        # 用yolov11对图像进行预测并保存裁剪之后的图像
        model = model_predict(
            model_path=model_path,
            image_dir=original_picture_path,
            out_dir=out_dir,
        )
        detect_results = model.predict()
        crop_results = model.crop_image(detect_results)
        # 用paddleocr对截取的图像进行字符提取
        ocr = paddleorc_predict(
            crop_result_dir=crop_picture_path,
            ocr_output_dir=ocr_picture_path,
        )
        ocr_results = ocr.predict()

        st.subheader("抽取车牌号结果")
        for ocr_result in ocr_results:
            st.write(ocr_result)

