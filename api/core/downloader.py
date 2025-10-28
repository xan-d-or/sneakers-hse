import os, hashlib, pathlib, mimetypes, requests
from api.core.rate_limiter import TokenBucket
from api.core.retry_utils import retry_request

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def download_image(url: str, out_dir: str, rl: TokenBucket, min_size=10_000) -> str | None:
    """Скачивает изображение с учётом rate-limit, возвращает путь к файлу или None."""
    rl.take()
    def _do():
        return requests.get(url, timeout=20)
    resp = retry_request(_do)
    if resp.status_code != 200:
        return None
    data = resp.content
    if len(data) < min_size:
        return None
    digest = sha256_bytes(data)
    ext = mimetypes.guess_extension(resp.headers.get("content-type", ""), strict=False) or ".jpg"
    path = pathlib.Path(out_dir) / f"{digest}{ext}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    return str(path)
