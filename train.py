from v11.ultralytics import YOLO

if __name__ == '__main__':
    """
    YOLOv11n的训练代码
    """
    model = YOLO(model="v11/ultralytics/cfg/models/11/yolo11n.yaml")
    model.info()
    train_result = model.train(
        data='datasets/VehicleLicense_data/data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device='cpu',
    )
    # 评估模型在验证集上的性能
    # model.val()
    # metrics = model.val()
    # print(metrics)




