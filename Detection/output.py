from ultralytics import YOLO
import os

def predict_image(model_path, images_dir, output_dir="output"):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in os.listdir(images_dir):

        img_path = os.path.join(images_dir, img_name)
        results = model.predict(img_path, save=True, project=output_dir, name=f"pred_{os.path.splitext(img_name)[0]}")
        
        print(f"Обработано: {img_name}")

if __name__ == "__main__":
    predict_image(
        model_path="yolo_100_dataset_v3_for_test.pt",
        images_dir="rostelekom\\datasets\\dataset\\test\\images",
    )