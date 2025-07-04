import os
import torch
import os
import torch
from ultralytics import YOLO
import os
from ultralytics import settings
from dotenv import load_dotenv
import gc

load_dotenv()

def main():
    torch.cuda.empty_cache()
    # Load a YOLO model
    model = YOLO("yolo11m.pt")
    print(model.info())
    settings.update({"wandb": True}) 


    # Train and Fine-Tune the Model
    model.train(project="info_model", 
                data='rostelekom\datasets\data_full.yml', 
                epochs=50, 
                batch = 16,  
                imgsz=640,
                workers = 1) 
    
    model.save('yolo_100_dataset_v3_for_test.pt')
    
if __name__ == '__main__':
    main()
