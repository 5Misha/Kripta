import os
import cv2

def resize_image(image, max_width=1200, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width/w, max_height/h)
    return cv2.resize(image, (int(w*scale), int(h*scale)))

def show_annotated_images(dataset_path, target_class, class_mapping=None):
    """
    Показывает изображения с bounding boxes для указанного класса.

    :param dataset_path: Путь к папке dataset (содержит train, val, test)
    :param target_class: Целевой класс (число, соответствующее классу в YOLO-формате)
    :param class_mapping: Словарь для преобразования class_id в имена классов
    """
    splits = ['train', 'val', 'test']
    color = (0, 255, 0)  # Зеленый цвет для bounding boxes
    thickness = 2

    for split in splits:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        images_dir = os.path.join(dataset_path, split, 'images')

        if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
            print(f"Папки в {split} не найдены. Пропускаем.")
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue

            label_path = os.path.join(labels_dir, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_file)

            if not os.path.exists(image_path):
                image_file = label_file.replace('.txt', '.png')
                image_path = os.path.join(images_dir, image_file)
                if not os.path.exists(image_path):
                    print(f"Изображение для {label_file} не найдено")
                    continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Ошибка загрузки: {image_path}")
                continue

            height, width = image.shape[:2]
            objects_found = False

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    class_id = int(parts[0])
                    if class_id != target_class:
                        continue

                    objects_found = True
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height

                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)

                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                    
                    if class_mapping:
                        class_name = class_mapping.get(class_id, str(class_id))
                        cv2.putText(image, 
                                   class_name,
                                   (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.9, color, thickness)

            if objects_found:
                window_name = f"{split}/{image_file}"
                image = resize_image(image)
                cv2.imshow(window_name, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

# Пример использования
dataset_path = "rostelekom/datasets/dataset"
target_class = 17 # Укажите нужный класс

# Опционально: словарь для подписей классов
class_mapping = {
  17: '21. Сплит система (наружный блок)'
}

show_annotated_images(dataset_path, target_class, class_mapping)
