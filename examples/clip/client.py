import asyncio
import base64
import csv
import io
import os
import statistics
import time
from pathlib import Path

import aiohttp
import numpy as np
import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

SERVE_URL = "http://127.0.0.1:8000/"

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_datasets_cache"

# write stats
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = DATA_DIR / "runtime_summary.csv"

if not SUMMARY_PATH.exists():
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "qps_target",
                "duration_s",
                "expected_requests",
                "completed_requests",
                "success_rate",
                "achieved_qps",
                "p50_s",
                "p90_s",
                "p99_s",
            ]
        )


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def food101_prompt(label: int, label_names) -> str:
    return f"a photo of {label_names[label].replace('_', ' ')}"


dataset = load_dataset(
    "ethz/food101",
    split="validation",
)
label_names = dataset.features["label"].names

NUM_SAMPLES = 64  # 10500


def prepare_payloads():
    payloads = []
    for i in tqdm(range(NUM_SAMPLES)):
        item = dataset[i]
        try:
            payloads.append(
                {
                    "image": pil_to_base64(item["image"]),
                    "text": food101_prompt(int(item["label"]), label_names),
                }
            )
        except Exception:
            print("encode failed")
            continue
    return payloads


async def send_request(session, payload, sem, results):
    async with sem:
        start = time.perf_counter()
        try:
            async with session.post(SERVE_URL, json=payload) as resp:
                await resp.read()
                end = time.perf_counter()
                latency = end - start
                results.append({"latency": latency, "ok": resp.status == 200, "start": start, "end": end})
        except Exception:
            end = time.perf_counter()
            results.append({"latency": None, "ok": False, "start": start, "end": end})


async def run_qps_stage(qps, duration, payloads, max_in_flight=1024):
    results = []
    sem = asyncio.Semaphore(max_in_flight)

    timeout = aiohttp.ClientTimeout(total=60)
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


async def main():
    print("Pre-generating payloads...")
    payloads = prepare_payloads()
    print(f"Prepared {NUM_SAMPLES} payloads")

    qps_schedule = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    duration = 5

    for qps in qps_schedule:
        print(f"\nRunning {qps} QPS for {duration}s...")
        results = await run_qps_stage(qps, duration, payloads)

        # compute metrics
        latencies = [r["latency"] for r in results if r["latency"] is not None]
        n_ok = sum(1 for r in results if r["ok"])
        n_total = len(results)

        start_all = min(r["start"] for r in results)
        end_all = max(r["end"] for r in results)
        total_time = end_all - start_all

        achieved_qps = n_total / total_time
        success_qps = n_ok / total_time
        success_rate = n_ok / n_total

        expected = qps * duration
        actual = n_total

        p50 = statistics.median(latencies)
        p90 = np.percentile(latencies, 90)
        p99 = np.percentile(latencies, 99)

        print(f"  expected≈{expected}, completed={actual}, wall={total_time:.3f}s")
        print(f"  achieved_qps={achieved_qps:.2f}, success_qps={success_qps:.2f}, success={success_rate:.2%}")
        print(f"  p50={p50:.3f}s  p90={p90:.3f}s  p99={p99:.3f}s")

        # write stats
        results_path = DATA_DIR / f"results_qps_{qps}.csv"
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["latency_s", "ok", "start", "end"])
            for r in results:
                writer.writerow(
                    [
                        r["latency"],
                        r["ok"],
                        r["start"],
                        r["end"],
                    ]
                )

        with open(SUMMARY_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    qps,
                    duration,
                    expected,
                    actual,
                    success_rate,
                    achieved_qps,
                    p50,
                    p90,
                    p99,
                ]
            )


if __name__ == "__main__":
    asyncio.run(main())
