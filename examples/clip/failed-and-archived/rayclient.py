import asyncio
import aiohttp

URL = "http://127.0.0.1:8000/clip"  # change if needed


async def send_request(session, image_path: str, text: str):
    with open(image_path, "rb") as img:
        form = aiohttp.FormData()
        form.add_field("image", img, filename="image.png", content_type="image/png")
        form.add_field("text", text)

        async with session.post(URL, data=form) as resp:
            resp.raise_for_status()
            return await resp.json()


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(128):
            tasks.append(send_request(session, "cat.png", "a cute cat"))
            await asyncio.sleep(0.05)

        # Run all requests concurrently
        results = await asyncio.gather(*tasks)

        for r in results:
            print(r)


if __name__ == "__main__":
    asyncio.run(main())
