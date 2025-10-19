import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats

def analyze_background_uniformity(image_path):
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
        
        # Конвертируем в HSV для анализа насыщенности и яркости
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Получаем размеры изображения
        height, width = img.shape[:2]
        
        # Анализируем края изображения (предполагая, что фон по краям)
        border_width = int(0.05 * min(height, width))  # 15% от меньшей стороны
        
        # Создаем маску для краев
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (width, height), 255, border_width)
        cv2.rectangle(mask, (border_width, border_width), 
                     (width - border_width, height - border_width), 0, -1)
        
        # Применяем маску для выделения фона
        background_pixels = img_rgb[mask == 255]
        
        if len(background_pixels) == 0:
            return None
        
        # Метрика 1: Средняя яркость фона
        background_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        background_brightness = np.mean(background_gray[mask == 255])
        
        # Метрика 2: Стандартное отклонение цвета фона (однородность)
        background_std = np.std(background_pixels, axis=0)
        color_uniformity = 1 / (1 + np.mean(background_std))  # Чем ближе к 1, тем однороднее
        
        # Метрика 3: Процент "белых" пикселей в фоне (V > 200 в HSV)
        background_v = img_hsv[:,:,2][mask == 255]
        white_pixels_ratio = np.sum(background_v > 200) / len(background_v)
        
        # Метрика 4: Процент "серых" пикселей в фоне (S < 50 и 100 < V < 200)
        background_s = img_hsv[:,:,1][mask == 255]
        gray_pixels_ratio = np.sum((background_s < 50) & (background_v > 100) & (background_v < 200)) / len(background_v)
        
        # Метрика 5: Энтропия фона (мера разнообразия)
        background_entropy = calculate_entropy(background_gray[mask == 255])
        
        # Метрика 6: Количество цветовых кластеров в фоне
        color_clusters = count_color_clusters(background_pixels)
        
        # Композитная оценка однородности фона
        uniformity_score = (color_uniformity + white_pixels_ratio + gray_pixels_ratio) / 3
        
        return {
            'brightness': float(background_brightness),
            'color_uniformity': float(color_uniformity),
            'white_ratio': float(white_pixels_ratio),
            'gray_ratio': float(gray_pixels_ratio),
            'entropy': float(background_entropy),
            'color_clusters': int(color_clusters),
            'uniformity_score': float(uniformity_score),
            'background_type': classify_background_type(uniformity_score, white_pixels_ratio, gray_pixels_ratio)
        }
        
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {str(e)}")
        return None

def calculate_entropy(pixels):
    """Вычисляет энтропию изображения как меру информационной насыщенности."""
    histogram = cv2.calcHist([pixels], [0], None, [256], [0, 256])
    histogram = histogram / histogram.sum()
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
    return entropy

def count_color_clusters(pixels, max_clusters=5):
    """Оценивает количество цветовых кластеров в фоне с помощью K-means."""
    if len(pixels) < max_clusters:
        return 1
    
    pixels_float = pixels.astype(np.float32)
    
    # Критерий остановки
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
    # Пробуем разное количество кластеров и выбираем оптимальное по WCSS
    wcss = []
    for k in range(1, min(6, len(pixels))):
        _, _, centers = cv2.kmeans(pixels_float, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        wcss.append(np.sum((pixels_float - centers[_.flatten()]) ** 2))
    
    # Находим "локоть" в графике WCSS
    optimal_k = find_elbow_point(wcss)
    return optimal_k

def find_elbow_point(wcss):
    """Находит точку 'локтя' в графике WCSS для определения оптимального числа кластеров."""
    if len(wcss) <= 2:
        return 1
    
    # Вычисляем вторые производные
    first_deriv = np.diff(wcss)
    second_deriv = np.diff(first_deriv)
    
    if len(second_deriv) == 0:
        return 1
    
    # Находим точку максимальной кривизны
    elbow_point = np.argmax(np.abs(second_deriv)) + 2
    return min(elbow_point, len(wcss))

def classify_background_type(uniformity_score, white_ratio, gray_ratio):
    """Классифицирует тип фона на основе вычисленных метрик."""
    if uniformity_score > 0.7 and white_ratio > 0.6:
        return "white_uniform"
    elif uniformity_score > 0.7 and gray_ratio > 0.4:
        return "gray_uniform"
    elif uniformity_score > 0.6:
        return "uniform"
    elif uniformity_score < 0.3:
        return "diverse"
    else:
        return "mixed"

def add_background_analysis_to_dataframe(df, images_base_path=""):
    """
    Добавляет анализ фона для всех изображений в DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame с колонкой 'path'
    images_base_path (str): Базовый путь к папке с изображениями
    
    Returns:
    pd.DataFrame: DataFrame с добавленными колонками анализа фона
    """
    background_data = []
    
    for idx, row in df.iterrows():
        image_path = Path(images_base_path) / row['path']
        
        result = analyze_background_uniformity(image_path)
        
        if result is not None:
            background_data.append(result)
        else:
            # Добавляем значения по умолчанию для пропущенных изображений
            background_data.append({
                'brightness': None,
                'color_uniformity': None,
                'white_ratio': None,
                'gray_ratio': None,
                'entropy': None,
                'color_clusters': None,
                'uniformity_score': None,
                'background_type': 'unknown'
            })
        
        # if (idx + 1) % 100 == 0:
        #     print(f"Обработано {idx + 1} изображений")
    
    # Создаем DataFrame с результатами
    background_df = pd.DataFrame(background_data)
    # print(background_df)
    # Объединяем с исходным DataFrame
    result_df = df.copy()
    result_df[['brightness', 'color_uniformity', 'white_ratio',
               'gray_ratio', 'entropy', 'color_clusters', 'uniformity_score', 
               'background_type']] = np.array(background_df)
    # pd.concat([df, background_df], axis=1)
    
    return result_df

# Визуализация результатов
def plot_background_analysis(df):
    """Визуализирует результаты анализа фона."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Распределение типов фона
    background_types = df['background_type'].value_counts()
    axes[0, 0].pie(background_types.values, labels=background_types.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Распределение типов фона')
    
    # Распределение оценки однородности
    axes[0, 1].hist(df['uniformity_score'].dropna(), bins=20, alpha=0.7)
    axes[0, 1].set_xlabel('Оценка однородности')
    axes[0, 1].set_ylabel('Количество')
    axes[0, 1].set_title('Распределение оценки однородности')
    
    # Соотношение белого/серого фона
    axes[0, 2].scatter(df['white_ratio'], df['gray_ratio'], alpha=0.5)
    axes[0, 2].set_xlabel('Доля белого фона')
    axes[0, 2].set_ylabel('Доля серого фона')
    axes[0, 2].set_title('Соотношение белого и серого фона')
    
    # Яркость vs однородность
    axes[1, 0].scatter(df['brightness'], df['uniformity_score'], alpha=0.5)
    axes[1, 0].set_xlabel('Яркость фона')
    axes[1, 0].set_ylabel('Оценка однородности')
    axes[1, 0].set_title('Яркость vs однородность')
    
    # Энтропия vs количество кластеров
    axes[1, 1].scatter(df['entropy'], df['color_clusters'], alpha=0.5)
    axes[1, 1].set_xlabel('Энтропия фона')
    axes[1, 1].set_ylabel('Количество цветовых кластеров')
    axes[1, 1].set_title('Энтропия vs сложность фона')
    
    # Распределение яркости
    axes[1, 2].hist(df['brightness'].dropna(), bins=20, alpha=0.7)
    axes[1, 2].set_xlabel('Яркость фона')
    axes[1, 2].set_ylabel('Количество')
    axes[1, 2].set_title('Распределение яркости фона')
    
    plt.tight_layout()
    plt.show()
