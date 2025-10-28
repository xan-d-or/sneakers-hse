import time, requests

def retry_request(func, max_retries=5, base_delay=0.3):
    """Ретраи с экспоненциальным backoff + jitter."""
    for i in range(max_retries):
        try:
            return func()
        except requests.RequestException as e:
            print(f"[retry {i}] {e}")
            time.sleep(base_delay * (2 ** i) + 0.05 * i)
    raise RuntimeError("Max retries exceeded")
