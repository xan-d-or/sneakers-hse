import numpy as np
import os
import pandas as pd
from rapidfuzz import fuzz

def directory_to_dataframe(path_to_dataset: str):

    # Список для хранения данных
    data = []

    # Проходим по всем подпапкам и файлам внутри них
    for root, dirs, files in os.walk(path_to_dataset):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                # Полный путь до файла
                full_path = os.path.join(root, file)
                
                # Относительный путь (относительно корневой папки)
                relative_path = os.path.relpath(full_path, path_to_dataset)
                
                # Имя класса — это имя подпапки (первая часть относительного пути)
                sneaker_class = relative_path.split(os.sep)[0]
                
                # Добавляем запись в список
                data.append({
                    "path": relative_path.replace("\\", "/"),
                    "sneaker_class": sneaker_class
                })

    # Создаём DataFrame
    df = pd.DataFrame(data)
    return df

def sneaker_class_to_brand(unique_classes: np.ndarray
                           ) -> dict[str, str]:
    BRANDS = ['adidas', 'asics', 'converse', 'new_balance',
              'nike', 'puma', 'reebok', 'salomon', 'vans', 'yeezy']

    # Порог похожести (можно варьировать, 70–85 обычно ок)
    SIMILARITY_THRESHOLD = 95

    def find_brand(sneaker_class):
        for brand in BRANDS:
            score = fuzz.partial_ratio(sneaker_class, brand)
            if score > SIMILARITY_THRESHOLD:
                return brand
        # Если не найдено — добавляем как новый "бренд"
        return sneaker_class

    # Определяем базовый бренд для каждого класса
    class_to_brand = {}
    for c in unique_classes:
        class_to_brand[c] = find_brand(c)
    return class_to_brand