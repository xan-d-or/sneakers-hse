import time

class TokenBucket:
    """Ограничивает скорость запросов (requests per second)."""
    def __init__(self, rps: float, burst: int):
        self.rps = rps
        self.capacity = burst
        self.tokens = burst
        self.last_time = time.time()

    def take(self):
        """Блокирует выполнение, если превышен лимит RPS."""
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_time) * self.rps)
        self.last_time = now
        if self.tokens < 1:
            sleep_for = (1 - self.tokens) / self.rps
            time.sleep(max(0, sleep_for))
            self.tokens = 0
        else:
            self.tokens -= 1
