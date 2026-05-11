from ultralytics import YOLO
from PIL import Image


class YOLODetector:
    def __init__(self, path_to_pt_file: str) -> None:
        self.model = YOLO(path_to_pt_file)
        self.SHOE_CLS = 3
        self.CLOTHING_CLS = 2
        self.conf = 0.2
        
    @staticmethod
    def crop_bbox(idx, box, result):
        # В этой функции помог перплексити
        # Координаты bbox [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[idx].cpu().numpy().astype(int)

        # Вырезать crop с padding (10% отступы)
        h, w = result.orig_shape
        pad = 0.1
        x1_pad = max(0, int(x1 - pad * (x2 - x1)))
        y1_pad = max(0, int(y1 - pad * (y2 - y1)))
        x2_pad = min(w, int(x2 + pad * (x2 - x1)))
        y2_pad = min(h, int(y2 + pad * (y2 - y1)))

        crop = result.orig_img[y1_pad:y2_pad, x1_pad:x2_pad]
        return crop

    def detect(self, img: Image.Image):
        '''
        1. Если на изображении есть класс shoes, то оставляем только 
        внутренность соответствующих ббоксов (обработка кейса картинок с ногами).
        2. Если на изображении нет класса shoes, но выделился единственный 
        bbox класса clothing, то берём его (обработка кейса стоковых картинок без ног). 
        Иногда это приводит к тому, что на картинке с ногами остаются только ноги, 
        но таких кейсов меньшинство.
        3. В противном случае оставляем исходную картинку.
        '''
        boxes = {}
        result = self.model.predict(img, conf=self.conf, verbose=False)[0]
        box = result.boxes
        num_boxes = box.shape[0]
        if num_boxes == 0:
            boxes['orig'] = result.orig_img
            return boxes
        has_shoe_cls = (box.cls.numpy() == self.SHOE_CLS).max()
        if has_shoe_cls:
            for i, cls in enumerate(box.cls.numpy()):
                if cls == self.SHOE_CLS:
                    boxes[f'shoe_{i}'] = self.crop_bbox(i, box, result)
        elif num_boxes == 1 and box.cls.numpy()[0] == self.CLOTHING_CLS:
            boxes[f'clothing_0'] = self.crop_bbox(0, box, result)
        else:
            boxes['orig'] = result.orig_img  
        return boxes