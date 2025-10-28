from abc import ABC, abstractmethod
from api.core.rate_limiter import TokenBucket
from api.core.retry_utils import retry_request
import json, time

class BaseParser(ABC):
    """Базовый класс для всех коннекторов."""
    source_name: str = "base"
    rps: float = 2
    burst: int = 4
    max_retries: int = 5

    def __init__(self):
        self.rl = TokenBucket(self.rps, self.burst)

    @abstractmethod
    def discover(self):
        """Итератор ссылок или ID товаров."""
        pass

    @abstractmethod
    def fetch(self, refs):
        """Итератор сырых объектов."""
        pass

    @abstractmethod
    def transform(self, raw_items):
        """Приведение к общей схеме NormalizedProduct."""
        pass

    def run(self, limit: int = 20):
        """Запускает весь цикл: discover->fetch->transform."""
        refs = list(self.discover())[:limit]
        raws = list(self.fetch(refs))
        products = list(self.transform(raws))
        print(f"[{self.source_name}] собрал {len(products)} товаров")
        print(json.dumps(products[:2], ensure_ascii=False, indent=2))
        return products
