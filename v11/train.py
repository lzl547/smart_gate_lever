from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov11n.pt")
    train_result = model.train(
        data='datasets/VehicleLicense_data/data.yaml',
        epochs='100',
    )




