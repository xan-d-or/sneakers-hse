from __future__ import annotations
import re, json, time
import requests
from urllib.parse import urljoin
from xml.etree import ElementTree as ET
from typing import Iterable, Dict, Any, List, Generator, Optional

from api.core.base_parser import BaseParser
from api.core.rate_limiter import TokenBucket
from api.core.retry_utils import retry_request

UA = {"User-Agent": "Mozilla/5.0 (compatible; AlikeSneakersBot/1.0; +https://alike.example/bot)"}

def _safe_get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    def _do():
        return requests.get(url, headers=UA, timeout=timeout)
    try:
        r = retry_request(_do)
    except Exception:
        return None
    if r is None or r.status_code != 200:
        return None
    return r

def _extract_jsonld(html: str) -> Optional[dict]:
    for m in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, flags=re.DOTALL | re.IGNORECASE
    ):
        try:
            data = json.loads(m.group(1))
            # иногда список
            if isinstance(data, list):
                for it in data:
                    if isinstance(it, dict) and it.get("@type") in ("Product", "OfferCatalog"):
                        return it
            elif isinstance(data, dict) and data.get("@type") in ("Product", "OfferCatalog"):
                return data
        except json.JSONDecodeError:
            continue
    return None

class ShopifyParser(BaseParser):
    """
    Стратегии:
      A) /products.json
      B) /collections.json -> /collections/{id}/products.json
      C) /sitemap.xml -> /products/<handle>  (fetch через .js -> .json -> HTML+JSON-LD)
    """
    source_name = "shopify"
    rps = 2
    burst = 4
    max_retries = 5

    def __init__(self, base_url: str, rps: float | None = None, burst: int | None = None):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        if rps is not None or burst is not None:
            self.rl = TokenBucket(rps or self.rps, burst or self.burst)

    def _discover_via_products_json(self, limit=250, max_pages=40) -> Generator[Dict[str, Any], None, None]:
        for page in range(1, max_pages + 1):
            self.rl.take()
            url = f"{self.base_url}/products.json?limit={limit}&page={page}"
            r = _safe_get(url)
            if not r:
                break
            payload = r.json()
            items = payload.get("products", [])
            if not items:
                break
            for p in items:
                yield {"id": p.get("id"), "handle": p.get("handle")}
            if len(items) < limit:
                break
            time.sleep(0.05)

    def _discover_via_collections(self, limit=250, max_pages=40) -> Generator[Dict[str, Any], None, None]:
        self.rl.take()
        rc = _safe_get(f"{self.base_url}/collections.json?limit=250")
        if not rc:
            return
        cols = rc.json().get("collections", [])
        for c in cols:
            cid = c.get("id")
            if not cid:
                continue
            for page in range(1, max_pages + 1):
                self.rl.take()
                url = f"{self.base_url}/collections/{cid}/products.json?limit={limit}&page={page}"
                r = _safe_get(url)
                if not r:
                    break
                items = r.json().get("products", [])
                if not items:
                    break
                for p in items:
                    yield {"id": p.get("id"), "handle": p.get("handle")}
                if len(items) < limit:
                    break
                time.sleep(0.05)

    def _discover_via_sitemap(self) -> Generator[Dict[str, Any], None, None]:
        self.rl.take()
        r = _safe_get(f"{self.base_url}/sitemap.xml")
        if not r:
            return
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError:
            return
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for loc in root.findall(".//sm:url/sm:loc", ns):
            url = (loc.text or "").strip()
            if re.search(r"/products/[^/]+/?$", url):
                handle = url.rstrip("/").split("/")[-1]
                yield {"id": None, "handle": handle}

    def discover(self) -> Iterable[Dict[str, Any]]:

        first = list(self._discover_via_products_json(limit=250, max_pages=1))
        if first:
            yield from first
            yield from self._discover_via_products_json(limit=250, max_pages=40)
            return
        
        got = False
        for item in self._discover_via_collections():
            got = True
            yield item
        if got:
            return
        
        for item in self._discover_via_sitemap():
            yield item

    def _fetch_one(self, handle: str) -> Optional[Dict[str, Any]]:
        """Возвращает унифицированный raw-словарь вида {"source":"js|json|html", "payload":..., "handle":...}"""
        
        self.rl.take()
        r = _safe_get(f"{self.base_url}/products/{handle}.js")
        if r:
            try:
                data = r.json()
                return {"source": "js", "handle": handle, "payload": data}
            except json.JSONDecodeError:
                pass

        self.rl.take()
        r = _safe_get(f"{self.base_url}/products/{handle}.json")
        if r:
            try:
                data = r.json()
                return {"source": "json", "handle": handle, "payload": data}
            except json.JSONDecodeError:
                pass

        self.rl.take()
        rh = _safe_get(f"{self.base_url}/products/{handle}")
        if rh:
            data = _extract_jsonld(rh.text)
            if data:
                return {"source": "html", "handle": handle, "payload": data}

        return None

    def fetch(self, refs: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for r in refs:
            handle = r.get("handle")
            if not handle:
                continue
            raw = self._fetch_one(handle)
            if raw:
                yield raw

    def transform(self, raw_items: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for raw in raw_items:
            src = raw["source"]
            handle = raw["handle"]

            if src == "json":
                p = raw["payload"].get("product", {})
                images = [img.get("src") for img in p.get("images", []) if img.get("src")]
                variants = p.get("variants", [])
                prices = []
                sku = None
                for v in variants:
                    if v.get("price") is not None:
                        try:
                            prices.append(float(v["price"]))
                        except Exception:
                            pass
                    if not sku and v.get("sku"):
                        sku = v["sku"]
                price_min = min(prices) if prices else None
                price_max = max(prices) if prices else None

                yield {
                    "source": self.source_name,
                    "source_product_id": p.get("id"),
                    "title": p.get("title"),
                    "brand": p.get("vendor"),
                    "model": p.get("title"),
                    "colorway": None,
                    "sku": sku,
                    "price": price_min,
                    "price_max": price_max,
                    "currency": None,  # Shopify JSON не всегда отдаёт валюту без Storefront API
                    "product_url": f"{self.base_url}/products/{handle}",
                    "image_urls": [f'https:{image}' for image in images],
                    # "image_urls": images,
                    # "payload_raw": raw,
                }
                continue

            if src == "js":
                p = raw["payload"]
                images = [img for img in p.get("images", []) if isinstance(img, str)]
                variants = p.get("variants", [])
                prices = []
                sku = None
                for v in variants:
                    pr = v.get("price")
                    if pr is not None:
                        try:
                            prices.append(float(pr))
                        except Exception:
                            pass
                    if not sku and v.get("sku"):
                        sku = v["sku"]
                price_min = min(prices) if prices else None
                price_max = max(prices) if prices else None

                yield {
                    "source": self.source_name,
                    "source_product_id": p.get("id"),
                    "title": p.get("title"),
                    "brand": p.get("vendor"),
                    "model": p.get("title"),
                    "colorway": None,
                    "sku": sku,
                    "price": price_min,
                    "price_max": price_max,
                    "currency": None,
                    "product_url": f"{self.base_url}/products/{handle}",
                    "image_urls": [f'https:{image}' for image in images],
                    # "image_urls": images,
                    # "payload_raw": raw,
                }
                continue

            if src == "html":
                d = raw["payload"]
                # images
                imgs: List[str] = []
                image_field = d.get("image")
                if isinstance(image_field, list):
                    imgs = [x for x in image_field if isinstance(x, str)]
                elif isinstance(image_field, str):
                    imgs = [image_field]
                # brand
                brand = None
                if isinstance(d.get("brand"), dict):
                    brand = d["brand"].get("name")
                elif isinstance(d.get("brand"), str):
                    brand = d["brand"]
                # prices
                prices = []
                offers = d.get("offers")
                if isinstance(offers, dict):
                    pr = offers.get("price")
                    if pr is not None:
                        try:
                            prices.append(float(pr))
                        except Exception:
                            pass
                elif isinstance(offers, list):
                    for o in offers:
                        pr = o.get("price")
                        if pr is not None:
                            try:
                                prices.append(float(pr))
                            except Exception:
                                pass
                price_min = min(prices) if prices else None
                price_max = max(prices) if prices else None

                yield {
                    "source": self.source_name,
                    "source_product_id": None,
                    "title": d.get("name"),
                    "brand": brand,
                    "model": d.get("name"),
                    "colorway": None,
                    "sku": None,
                    "price": price_min,
                    "price_max": price_max,
                    "currency": d.get("offers", {}).get("priceCurrency") if isinstance(d.get("offers"), dict) else None,
                    "product_url": f"{self.base_url}/products/{handle}",
                    "image_urls": [f'https:{image}' for image in images],
                    # "image_urls": images,
                    # "payload_raw": raw,
                }
