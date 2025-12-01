from v11.ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2 as cv
from pprint import pprint
import json
import streamlit as st

def init_ocr():
    return PaddleOCR(use_angle_cls=True, lang='ch')

if __name__ == '__main__':
    model_path = "runs/detect/train/weights/best.pt"
    IMAGES_DIR = r"F:\ProjectForWork\vehicleLicenseRecognition\test_picture"  # 你的图片文件夹
    OUT_DIR = r"F:\ProjectForWork\vehicleLicenseRecognition\out_results"
    CONF_THRES = 0.25
    PAD_RATIO = 0.06
    model = YOLO(model=model_path)
    """
    预测整个文件夹的图片，返回ultralytics.engine.results.Results object列表
    Results的attributes有：
        boxes:目标检测任务的信息，包含检测出的所有bounding boxes信息
            cls:类别列表
            conf:置信度列表
            data:[[xyxy坐标 + 置信度 + 类别],
                  [xyxy坐标 + 置信度 + 类别],
                  ...                      ]
            xyxy,xyxyn:左上 + 右下坐标，归一化之后的坐标
            xywh,xywhn:中心点 + 宽高，归一化之后的坐标
        keypoints:关键点检测任务的信息
        masks:分割任务的信息
        names:类别词典
        obb:旋转框检测任务信息
        orig_img:输入的原图数组
        orig_shape:输入图像的尺寸
        ......
    """
    # result = model.predict(source='./test_picture',
    #                        save=True,
    #                        iou=0.4,
    #                        conf=0.4,
    #                        device='cpu'
    #                        )
    # pprint(result)
    # print("*" * 100)
    test_picture_path = r'./test_picture/1.jpg'
    # 预测图片输出，置信度不低于0.4，iou不高于0.4
    result = model.predict(source=test_picture_path,
                           save=True,
                           iou=0.4,
                           conf=0.4,
                           device='cpu'
                           )
    pprint(result)
    print(result[0].boxes)    #打印输出结果的boxes
    print("*" * 1000)
    print(result[0].boxes.xywh)
    img = result[0].orig_img
    print(img.shape)
    print(img)
    # 把坐标提取出来并在原图上裁剪
    for box in result[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
        crop = img[y1:y2, x1:x2]
        cv.imwrite("plate.jpg", crop)

    print("*" * 1000)
    ocr = init_ocr()
    res = ocr.predict('plate.jpg')
    print(type(res))
    pprint(res)
    print("*" * 1000)
    print(res[0]['doc_preprocessor_res']['output_img'])

    # opencv图像显示
    cv.namedWindow('output_img', cv.WINDOW_NORMAL)
    cv.imshow('output_img', res[0]['doc_preprocessor_res']['output_img'])

    # print(res[0]['rec_texts'])

    cv.waitKey(0)
    print("*" * 1000)
    # 等待按键，按任意键关闭窗口
    # print("按任意键关闭窗口...")
    # cv.waitKey(0)  # 程序停在这里，直到用户按键
    # cv.destroyAllWindows()  # 然后关闭所有OpenCV窗口
