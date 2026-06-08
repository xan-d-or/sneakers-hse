import asyncio
import os
import logging

import aiohttp
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.types import InputMediaPhoto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
API_URL = os.getenv("API_URL", "http://sneakers_hse_api:8000")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "👟 Привет! Пришли фото кроссовок — найду похожие в нашей базе."
    )


@dp.message(F.photo)
async def handle_photo(message: types.Message):
    await message.answer("🔍 Ищу похожие...")

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)

    try:
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field(
                "image",
                file_bytes.read(),
                filename="image.jpg",
                content_type="image/jpeg",
            )
            async with session.post(f"{API_URL}/search", data=form, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    await message.answer(f"❌ Ошибка API: {resp.status}")
                    return
                result = await resp.json()
    except Exception as e:
        logger.exception("API call failed")
        await message.answer(f"❌ Не удалось подключиться к сервису: {e}")
        return

    hits = result.get("results", [])
    if not hits or not hits[0].get("image_urls"):
        await message.answer("😔 Ничего похожего не нашлось.")
        return

    first = hits[0]
    urls = first["image_urls"][:5]
    ids = first["ids"][:5]
    distances = first["distances"][:5]

    # Send as media group (max 10 items)
    media = []
    for i, (url, img_id, dist) in enumerate(zip(urls, ids, distances)):
        label = img_id.split("/")[0]  # class name from path
        caption = f"#{i+1} {label}\n(dist: {dist:.3f})" if i == 0 else None
        media.append(InputMediaPhoto(media=url, caption=caption))

    try:
        await message.answer_media_group(media)
    except Exception as e:
        logger.exception("Failed to send media group")
        # Fallback: send URLs as text
        text = "\n".join(f"{i+1}. {img_id.split('/')[0]}" for i, img_id in enumerate(ids))
        await message.answer(f"Топ-5 похожих:\n{text}")

    latency = result.get("latency_ms", 0)
    await message.answer(f"⏱ {latency:.0f} мс")


@dp.message(~F.photo)
async def handle_other(message: types.Message):
    await message.answer("Пришли фото кроссовков 👟")


async def main():
    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
