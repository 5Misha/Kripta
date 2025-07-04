import os
import xml.etree.ElementTree as ET
import re
from pathlib import Path
import random
import shutil
from dotenv import load_dotenv
import yaml
load_dotenv()

# Создаем пути к папкам
MAIN_PATH = 'rostelekom/datasets'
path_to_annotations = os.path.join(MAIN_PATH, 'Annotations')
folder_with_images = os.path.join(MAIN_PATH, 'images')
folder_with_annotations = os.path.join(MAIN_PATH, 'YOLO')
destination_directory = os.path.join(MAIN_PATH, 'dataset')
data_yml = os.path.join(MAIN_PATH, 'data_full.yml')


# Функция для извлечения чисел и названия из строки
def extract_numbers_and_title(input_string: str):
    match = re.match(r'(\d+\.)\s*\.\s*(.*)', input_string)
    if match:
        numbers = match.group(1)
        title = match.group(2)
        return numbers, title
    else:
        return None, input_string

# Функция создания словаря классов
def create_class_mapping(xml_folder: str, selected_classes: list) -> dict:
    """
    Проходит по всем XML-файлам и создаёт словарь сопоставления классов,
    оставляя только уникальные классы.

    :param xml_folder: Путь к папке с XML-файлами.
    :param selected_classes: Список уникальных выбранных классов.
    :return: Словарь сопоставления {имя класса: индекс}.
    """
    class_counter = {}
    current_index = 0
    undefined_classes = dict()

    for filename in os.listdir(xml_folder): # перебор названий из папки в xml_folder
        if filename.endswith('.xml'):
            xml_file_path = os.path.join(xml_folder, filename) # путь к определенному файлу
            tree = ET.parse(xml_file_path) # парсим файл в вид дерева
            root = tree.getroot() # выделяем корневой узел дерева

            for obj in root.findall('object'): # перебираем название каждого объекта
                _, class_name = extract_numbers_and_title(obj.find('name').text) # берем только название без числа
                # Добавляем класс в словарь, только если он в selected_classes
                if class_name in selected_classes and class_name not in class_counter:
                    class_counter[class_name] = current_index
                    current_index += 1
                # иначе добавляем класс в несчитанные
                if class_name not in selected_classes:
                    try:
                        undefined_classes[class_name] += 1
                    except:
                        undefined_classes[class_name] = 1


    print('__________\nНеучтенные классы: ')
    for obj in undefined_classes:
        if undefined_classes[obj] > 0:
            print(obj, ' - ', undefined_classes[obj])
    print('\n___________')
    return class_counter

def create_yml_file(class_mapping, data_yml):
    '''
    class_mapping - результат работы функции create_class_mapping, которая возвращает словарь классов
    data_yml - путь для сохранения файла data_full формата yml
    '''
    # Преобразуем class_mapping в нужный формат
    names = {v: k for k, v in class_mapping.items()} # меняем местами ключи со значениями для формата YAML
    
    # Создаем структуру данных для YAML
    yaml_data = {
        'names': names,
        'path': destination_directory,
        'train': 'train',
        'val': 'val',
        }
    
    # Записываем данные в YAML файл с кавычками вокруг строк
    with open(data_yml, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, allow_unicode=True, default_flow_style=False, width=1024) 
        # allow_unicode=True - разрешаем Unicode-символы, default_flow_style=False - читаемый многострочный формат, width - макс ширина строки перед переносом

# Функция для чтения выбранных классов
def load_selected_classes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: # Явно указываем кодировку utf-8
        result = []
        for line in f:
            if len(line.strip()) :  # если строка не пустая
                result.append(line.strip())
        return result # список из обработанных строк

# Конвертация XML в YOLO с фильтрацией классов
def convert_xml_to_yolo_filtered(xml_file, class_mapping, selected_classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    yolo_annotations = []

    for obj in root.findall('object'):
        _, class_name = extract_numbers_and_title(obj.find('name').text)
        class_id = class_mapping.get(class_name)

        if class_name not in selected_classes:
            continue
        # Получение координат bounding box
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text) # левая граница
        ymin = int(bndbox.find('ymin').text) # верхняя граница
        xmax = int(bndbox.find('xmax').text) # правая граница
        ymax = int(bndbox.find('ymax').text) # нижняя граница
        
        # Делим на width и height для нормализации. Это помогает при изменении масштаба картинки
        x_center = (xmin + xmax) / 2 / width 
        y_center = (ymin + ymax) / 2 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height
        # Строку с аннотацией добавляем в список yolo_annotations
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    if yolo_annotations:
        yolo_file = os.path.join(folder_with_annotations, Path(filename).stem + '.txt')
        with open(yolo_file, 'w') as f:
            f.write("\n".join(yolo_annotations)) # каждый объект это новая строка


# Функция разделения на train/val
from collections import defaultdict
from typing import List, Tuple, Dict, Set

def parse_yolo_annotations(annotations_path: str) -> Dict[str, Set[int]]:
    """
    Парсит YOLO-аннотации и извлекает классы для каждого изображения.

    :param annotations_path: Путь к папке с YOLO-аннотациями (.txt файлы).
    :return: Словарь, где ключ - имя изображения (без расширения), значение - множество классов входящих в изображение.
    {'img1': {0, 1},  # содержит классы 0 и 1
    'img2': {1}      # содержит класс 1}
    """
    image_classes = {} # ключ - имя изображения, значение - кол-во объектов
    for file_name in os.listdir(annotations_path):
        if file_name.endswith('.txt'):
            image_name = os.path.splitext(file_name)[0] # удаляем расширение
            with open(os.path.join(annotations_path, file_name), 'r') as f:
                # в каждой не пустой строке берем первый элемент до пробела и делаем int-ом
                class_ids = {int(line.split()[0]) for line in f if line.strip()} 
            image_classes[image_name] = class_ids # ключ - имя файла, значение - номер
    return image_classes


def split_dataset_from_yolo(
    annotations_path: str, 
    image_files: List[str],
    split_ratios: Tuple[float, float, float] = (0.8, 0.2, 0.0), 
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Разделяет датасет на train, val и test с учетом YOLO-аннотаций.
    :param annotations_path: Путь к папке с YOLO-аннотациями (.txt файлы).
    :param image_files: Список путей к изображениям.
    :param split_ratios: Коэффициенты разбиения (train, val, test).
    :param seed: Случайное зерно для повторяемости.
    :return: Словарь с ключами 'train', 'val', 'test' и списками файлов.
    """
    random.seed(seed)
    assert sum(split_ratios) == 1.0, "Сумма коэффициентов разбиения должна быть равна 1."

    # Парсим классы из YOLO-аннотаций
    labels = parse_yolo_annotations(annotations_path)

    # Сопоставляем классы с полными путями к изображениям
    labeled_files = {os.path.splitext(os.path.basename(file))[0]: file for file in image_files}

    # Разделение по классам
    class_files = defaultdict(list)
    class_files_count = defaultdict(int)
    for image_name, classes in labels.items():
        if image_name in labeled_files:
            for class_id in classes:
                class_files[class_id].append(labeled_files[image_name])
                class_files_count[class_id] += 1
    # С помощью перебора каждого класса, выводим кол-во в нем изображений
    for class_id, count in class_files_count.items():
        print("Класс: ", class_id, " Количество изображений: ", count)
    # Уникальные файлы в каждом разделе
    splits = {'train': set(), 'val': set(), 'test': set()}

    # Разделяем данные для каждого класса
    for class_id, files in class_files.items():
        random.shuffle(files)
        n = len(files)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])

        splits['train'].update(files[:train_end])
        splits['val'].update(files[train_end:val_end])
        splits['test'].update(files[val_end:])

    # Убираем пересечения между наборами
    splits['val'] -= splits['train']
    splits['test'] -= (splits['train'] | splits['val'])

    return {key: list(files) for key, files in splits.items()}



def annotation_exists(image_path, annotations_folder):
    """
    Проверяет, существует ли аннотация для данного изображения.
    Добавлены отладочные сообщения для диагностики.
    
    :param image_path: Путь к изображению.
    :param annotations_folder: Путь к папке с аннотациями.
    :return: True, если аннотация существует, иначе False.
    """
    image_stem = Path(image_path).stem # берем из пути только название картинки
    annotation_path = os.path.join(annotations_folder, f"{image_stem}.txt") # ищем в аннотациях такое же название
    annotation_exists = os.path.exists(annotation_path) # проверяем есть ли такой файл

    return annotation_exists


# Функция копирования данных
def copy_images_with_annotations(images_list, split_type, annotations_folder, destination_directory):
    """
    Копирует изображения и соответствующие аннотации в папку train или val.
    Добавлены отладочные сообщения для диагностики.

    :param images_list: Список путей к изображениям.
    :param split_type: Тип разбиения ('train' или 'val').
    :param annotations_folder: Папка с аннотациями.
    :param destination_directory: Папка назначения для датасета.
    """
    images_dir = os.path.join(destination_directory, split_type, 'images')
    labels_dir = os.path.join(destination_directory, split_type, 'labels')

    os.makedirs(images_dir, exist_ok=True) # если папок нет, то создаем их
    os.makedirs(labels_dir, exist_ok=True)

    for image_path in images_list:
        if annotation_exists(image_path, annotations_folder): # совпадение аннотации и картинки
            # Копируем изображение
            shutil.copy(image_path, os.path.join(images_dir, os.path.basename(image_path)))

            # Копируем соответствующую аннотацию
            image_stem = Path(image_path).stem
            annotation_path = os.path.join(annotations_folder, f"{image_stem}.txt")
            shutil.copy(annotation_path, os.path.join(labels_dir, f"{image_stem}.txt"))


# Функция записи путей в .txt файлы
def write_file_list(files, output_path):
    with open(output_path, 'w') as f:
        for file in files:
            f.write(f"{os.path.abspath(file)}\n")


def clear_directory(directory):
    """
    Удаляет все файлы в указанной директории.

    :param directory: Путь к директории.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Удаляем папку со всем её содержимым
    os.makedirs(directory, exist_ok=True)  # Создаём пустую папку заново


clear_directory(folder_with_annotations)

clear_directory(os.path.join(destination_directory, 'train'))
clear_directory(os.path.join(destination_directory, 'val'))
clear_directory(os.path.join(destination_directory, 'test'))

# Создание необходимых папок
os.makedirs(folder_with_annotations, exist_ok=True)
os.makedirs(destination_directory, exist_ok=True)

# Основной блок выполнения
selected_classes_file = os.path.join(MAIN_PATH, 'selected_classes.txt')
selected_classes = load_selected_classes(selected_classes_file)
print(selected_classes)

class_mapping = create_class_mapping(path_to_annotations, selected_classes)
print(class_mapping)

# Переформатируем из XML в YAML
for xml_file in os.listdir(path_to_annotations): 
    if xml_file.endswith('.xml'):
        convert_xml_to_yolo_filtered(
            os.path.join(path_to_annotations, xml_file),
            class_mapping,
            selected_classes
        )

create_yml_file(class_mapping, data_yml) # создаем YAML файл
# Разделение данных
image_files = [os.path.join(folder_with_images, img) for img in os.listdir(folder_with_images)]
print(f'Всего фотографий:[{len(image_files)}] ')
splited_dataset = split_dataset_from_yolo(folder_with_annotations, image_files)
train = splited_dataset['train']
val = splited_dataset['val']
test = splited_dataset['test']

# Копирование данных
copy_images_with_annotations(train, 'train', folder_with_annotations, destination_directory)

copy_images_with_annotations(val, 'val', folder_with_annotations, destination_directory)

copy_images_with_annotations(test, 'test', folder_with_annotations, destination_directory)

write_file_list(train, MAIN_PATH + '/train.txt')
write_file_list(val, MAIN_PATH + '/val.txt')
write_file_list(test, MAIN_PATH + '/test.txt')


print("Данные успешно подготовлены!")

for dir in ['train', 'val', 'test']:

    #print(MAIN_PATH + "/dataset/" + dir + "/images")
    image_files = [Path(img).stem for img in os.listdir(MAIN_PATH + "/dataset/" + dir + "/images") if img.endswith('.jpeg') or img.endswith('.jpg')]
    label_files = [Path(lbl).stem for lbl in os.listdir(MAIN_PATH + "/dataset/" + dir + "/labels") if lbl.endswith('.txt')]

    # Найдём изображения без аннотаций
    missing_labels = [img for img in image_files if img not in label_files]

    # Найдём аннотации без изображений
    missing_images = [lbl for lbl in label_files if lbl not in image_files]

    print(f"Изображения без аннотаций в {dir}:", missing_labels)
    print(len(missing_labels))
    print(f"Аннотации без изображений в {dir}:", missing_images)

print(f"Количество изображений в train: {len(list(Path(MAIN_PATH + "/dataset/train/images").iterdir()))}")
print(f"Количество изображений в val: {len(list(Path(MAIN_PATH + "/dataset/val/images").iterdir()))}")
print(f"Количество изображений в test: {len(list(Path(MAIN_PATH + "/dataset/test/images").iterdir()))}")

print(MAIN_PATH)
