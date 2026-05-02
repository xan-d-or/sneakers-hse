import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats

def analyze_background_uniformity(image_path, border_margin):
    """
    Анализирует однородность фона изображения.
    Возвращает метрики, оценивающие насколько фон белый/серый и однородный.
    """
    try:
        # Загружаем изображение
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Конвертируем в RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        # Получаем размеры изображения
        height, width = img.shape[:2]
        
        # Анализируем края изображения (предполагая, что фон по краям)
        border_width = int(border_margin * min(height, width))  # 5% от меньшей стороны
        
        # Создаем маску для краев
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (width, height), 255, border_width)
        cv2.rectangle(mask, (border_width, border_width), 
                     (width - border_width, height - border_width), 0, -1)
        
        # Применяем маску для выделения фона
        background_pixels = img_rgb[mask == 255]
        
        if len(background_pixels) == 0:
            return None
        
        # Стандартное отклонение цвета фона (однородность)
        background_std = np.std(background_pixels, axis=0)
        background_uniformity = 1 / (1 + np.mean(background_std))  # Чем ближе к 1, тем однороднее
        
        return float(background_uniformity)
        
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {str(e)}")
        return None


def add_background_analysis_to_dataframe(df: pd.DataFrame,
                                         images_base_path: str = "",
                                         border_margin: float = 0.05):
    """
    Добавляет анализ фона для всех изображений в DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame с колонкой 'path'
    images_base_path (str): Базовый путь к папке с изображениями
    
    Returns:
    pd.DataFrame: DataFrame с колонкой однородности фона
    """
    background_data = []
    
    for idx, row in df.iterrows():
        image_path = Path(images_base_path) / row['path']
        result = analyze_background_uniformity(image_path, border_margin)
        background_data.append(result)
    
    result_df = df.copy()
    result_df['background_uniformity'] = background_data
    
    return result_df
