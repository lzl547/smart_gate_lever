from v11.ultralytics import YOLO
from pprint import pprint
import os
import cv2 as cv
import numpy as np

class model_predict():
    """
    预测图像中车牌位置并且裁剪
    """
    def __init__(self, model_path, image_dir, out_dir,
                 conf_thres=0.4, pad_ratio=0.06, iou_thres=0.4):
        self.model_path = model_path
        self.image_dir = image_dir
        self.out_dir = out_dir
        self.conf_thres = conf_thres
        self.pad_ratio = pad_ratio
        self.iou_thres = iou_thres

    def predict(self):
        """
        预测图片输出，并且保存在./out_results/predict中，置信度不低于0.4，iou不高于0.4
        :return: 将预测结果图像保存在out_dir中，返回预测结果图像列表
        """
        model = YOLO(model=self.model_path)
        results = model.predict(source=self.image_dir,
                                save=True,
                                iou=self.iou_thres,
                                conf=self.conf_thres,
                                device='cpu',
                                project=self.out_dir,
                                name='predict',
                                exist_ok=True,
                                )
        return results

    def crop_image(self, results):
        """
        按照预测框截取并填充图片，并且保存在./out_results/crop_results文件夹中
        :param results: yolov11的预测的结果
        :return: 按照预测框截取并填充的图片
        """
        folder_path = self.out_dir + r'/crop_results'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(folder_path, "已建立")

        crop_image_list = []
        img_name = 0
        for result in results:
            orig_img = result.orig_img
            print(orig_img.shape)
            idx = int(np.argmax(result.boxes.conf))
            x1, y1, x2, y2 = result.boxes.xyxy[idx].numpy()
            # 对预测框进行扩充裁剪
            img_h = len(orig_img[0])  # 图像高
            print("*" * 100, img_h)
            img_w = len(orig_img[1])  # 图像宽
            w = x2 - x1
            h = y2 - y1
            pad_w = w * self.pad_ratio
            pad_h = h * self.pad_ratio
            new_x1 = max(0, int(x1 - pad_w))  # 填充之后的坐标
            new_y1 = max(0, int(y1 - pad_h))
            new_x2 = min(img_w - 1, int(x2 + pad_w))
            new_y2 = min(img_h - 1, int(y2 + pad_h))
            crop = orig_img[new_y1:new_y2, new_x1:new_x2]
            crop_image_list.append(crop)
            cv.imwrite(f"{folder_path}/{img_name}.jpg", crop)
            img_name += 1

        return crop_image_list

if __name__ == '__main__':
    model_predict = model_predict(
        model_path="runs/detect/train/weights/best.pt",
        image_dir=r"F:\ProjectForWork\vehicleLicenseRecognition\test_picture",
        out_dir=r"F:\ProjectForWork\vehicleLicenseRecognition\out_results",
    )
    results = model_predict.predict()
    print("3" * 100)
    pprint(results[5])
    pprint(results[5].boxes)
    pprint(model_predict.crop_image(results))