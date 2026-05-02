import PIL
from PIL import Image
import imagehash


def calc_hash(img_path, hash_func):

    image = Image.open(img_path)

    hashed_img = hash_func(image)
    return hashed_img

def calc_hash_df(df, path_to_dataset, hash_func):
    # Список для хранения данных
    data = []

    # Проходим по всем подпапкам и файлам внутри них
    
    hashed_img = df['path'].apply(lambda x: calc_hash(path_to_dataset / x, hash_func))

    return hashed_img