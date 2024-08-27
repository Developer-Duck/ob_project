from ultralytics import YOLO

def main():
    # Load a pretrained YOLO model
    model = YOLO("yolov8n.pt")  # 사전 훈련된 YOLO 모델 로드

    # Define the path to your custom dataset YAML file
    data_config = "C:/Users/xxxfl/OneDrive/바탕 화면/동아리/bl_object_project/Finger-detection-1/data.yaml"
    


    # Train the model on your custom dataset
    model.train(data=data_config, epochs=50, imgsz=640)  # epochs와 image size는 필요에 따라 조정

    # Evaluate the model performance on the validation set
    metrics = model.val()
    print("Validation metrics:", metrics)

    # Predict on a new image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Index_finger_%3D_to_attention.JPG/640px-Index_finger_%3D_to_attention.JPG"
    results = model(image_url)

    # Display the results
    results.show()  # 이미지에 예측 결과를 시각화하여 표시
    results.save()  # 예측 결과를 파일로 저장

    # Export the trained model to ONNX format
    export_path = model.export(format="onnx")
    print(f"Model exported to {export_path}")

if __name__ == '__main__':
    main()
