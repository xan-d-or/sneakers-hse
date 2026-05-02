import os
import pandas as pd

def filter_dataset(
        df: pd.DataFrame,
        path_to_dataset: str,
        class_names_to_remove: list[str]=["yeezy_slide"],
        bad_images_md_path="../2-exploration/bad_images.md"
):
    """

    """
    # Вычищаем Yeezy Slide, т.к. они не укладываются в наши представления о кроссовках
    df = df[df['sneaker_class'].apply(lambda x: x not in class_names_to_remove)]
    # Фильтруем битые картинки, рисунки и т.п.
    # Плохие картинки мы отсмотрели вручную. Список можно увидеть в файле bad_images
    with open(bad_images_md_path) as f:
        bad_images = f.read().strip().split("\n")

    bad_image_paths = [
        image[image.find("(") + 1 : -1] for image in bad_images if len(image) > 0
    ]  # В md указаны пути в формате ![name](path). Поэтому путь это все, что находится в круглых скобках ()

    bad_image_paths = [
        os.path.relpath(
            image, start=path_to_dataset
        )  # Тут заменил, надо заменить пути в bad_images.md
        for image in bad_image_paths
    ]

    print(f"Отбросили изображений: {len(bad_image_paths)}")

    df = df[df["path"].apply(lambda x: x not in bad_image_paths)]
    
    print(f"Осталось изображений: {df.shape[0]}")

    return df