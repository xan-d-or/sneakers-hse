import os
import pandas as pd


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