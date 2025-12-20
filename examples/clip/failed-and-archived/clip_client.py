import asyncio
import aiohttp
import time
import io
import statistics
from tqdm import tqdm
import multiprocessing as mp


from examples.clip.dataset import get_dataset

mp.set_start_method("spawn", force=True)


# -----------------------------
# Pre-generation
# -----------------------------
def image_to_bytes(image, format="PNG"):
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def prepare_payloads(dataset):
    payloads = []
    for image, text in tqdm(dataset):
        payloads.append(
            {
                "image": image_to_bytes(image),
                "text": text,
            }
        )
    return payloads


dataset = get_dataset()


def encode_sample(idx):
    image, text = dataset[idx]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "image": buf.getvalue(),
        "text": text,
    }


def prepare_payloads_mp(num_workers=mp.cpu_count(), chunksize=64):
    n = 1024

    payloads = []

    with mp.Pool(processes=num_workers) as pool:
        for payload in tqdm(
            pool.imap_unordered(encode_sample, range(n), chunksize=chunksize),
            total=n,
        ):
            payloads.append(payload)

    return payloads


# -----------------------------
# Async request
# -----------------------------
async def send_request(session, payload, sem, results):
    async with sem:
        start = time.perf_counter()
        try:
            data = aiohttp.FormData()
            data.add_field(
                "image",
                payload["image"],
                filename="image.png",
                content_type="image/png",
            )
            data.add_field("text", payload["text"])

            async with session.post("http://localhost:8000/clip", data=data) as resp:
                await resp.read()
                latency = time.perf_counter() - start
                results.append(
                    {
                        "latency": latency,
                        "ok": resp.status == 200,
                    }
                )
        except Exception:
            results.append({"latency": None, "ok": False})


# -----------------------------
# QPS stage (absolute-time scheduler)
# -----------------------------
async def run_qps_stage(qps, duration, payloads, max_in_flight=1024):
    results = []
    sem = asyncio.Semaphore(max_in_flight)

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        end_time = start_time + duration
        interval = 1.0 / qps

        next_fire = start_time
        idx = 0
        tasks = []

        while next_fire < end_time:
            # wait until scheduled fire time
            await asyncio.sleep(max(0, next_fire - loop.time()))

            payload = payloads[idx % len(payloads)]
            tasks.append(asyncio.create_task(send_request(session, payload, sem, results)))

            idx += 1
            next_fire += interval

        await asyncio.gather(*tasks)

    return results


# -----------------------------
# Main
# -----------------------------
async def main():
    print("Pre-generating payloads...")
    payloads = prepare_payloads_mp()
    print(f"Prepared {len(payloads)} payloads")

    qps_schedule = [1, 2, 4, 8, 128, 256, 512, 1024, 2048]
    duration = 5

    for qps in qps_schedule:
        print(f"\nRunning {qps} QPS for {duration}s...")
        results = await run_qps_stage(qps, duration, payloads)

        latencies = [r["latency"] for r in results if r["latency"] is not None]
        success = sum(r["ok"] for r in results) / len(results)

        expected = qps * duration
        actual = len(results)

        p50 = statistics.median(latencies) if latencies else float("nan")

        print(f"  expected≈{expected}, actual={actual}, p50={p50:.3f}s, success={success:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
