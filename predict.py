from ultralytics import YOLO

model_path = r"runs/detect/train/weights/best.pt"  # 这里写训练出来的模型路径
image_path = r"F:\ProjectForWork\vehicleLicenseRecognition\test_picture"  # 这里是要预测的图片路径，可以是一个文件夹
model = YOLO(model_path)
result = model(source=image_path, device='cpu')

