"""轻量并发压测：打 /agent/stream 主聊天链路，记录延迟分位数（P50/P95/P99）。

用法（先把后端跑起来）：
    python benchmark.py                        # 默认 5 并发 x 20 请求
    python benchmark.py -c 10 -n 40             # 10 并发 x 40 请求
    python benchmark.py --url http://127.0.0.1:8000 -u customer -p customer123

说明：
- 每个请求独立 session_id（DST 状态互不干扰），问题从池里轮换取知识类问句，
  覆盖「意图识别 -> 检索 -> 重排 -> LLM 流式生成」全链路，是最重的真实路径。
- 只用 requests + 线程池，不引压测框架；单请求延迟 = 发出 -> 流式收满全部正文。
"""
import argparse
import concurrent.futures as futures
import random
import string
import sys
import time
from collections import defaultdict

import requests

QUESTION_POOL = [
    "手机的保修政策是什么样的",
    "退货的运费由谁承担",
    "退款多久能到账",
    "7天无理由退货是怎么规定的",
    "你们在售的手机有哪些型号",
]

def login(base: str, username: str, password: str) -> str:
    resp = requests.post(f"{base}/auth/login",
                         json={"username": username, "password": password}, timeout=15)
    resp.raise_for_status()
    return resp.json()["token"]


def one_request(base: str, token: str, question: str) -> tuple:
    """单个请求：返回 (耗时秒, 状态码, 响应字节数)。"""
    session_id = "bench-" + "".join(random.choices(string.ascii_lowercase, k=10))
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"question": question, "session_id": session_id}
    t0 = time.perf_counter()
    try:
        with requests.post(f"{base}/agent/stream", json=payload,
                           headers=headers, stream=True, timeout=120) as resp:
            body = b""
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    body += chunk
            return time.perf_counter() - t0, resp.status_code, len(body)
    except Exception as e:
        return time.perf_counter() - t0, -1, str(e)[:80]


def percentile(sorted_lat: list, p: float) -> float:
    if not sorted_lat:
        return float("nan")
    idx = min(int(len(sorted_lat) * p / 100), len(sorted_lat) - 1)
    return sorted_lat[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("-c", "--concurrency", type=int, default=5)
    ap.add_argument("-n", "--requests", type=int, default=20)
    ap.add_argument("-u", "--username", default="customer")
    ap.add_argument("-p", "--password", default="customer123")
    args = ap.parse_args()

    print(f"压测目标 {args.url} | 并发 {args.concurrency} | 请求 {args.requests}")
    token = login(args.url, args.username, args.password)
    print("登录成功，开始压测...\n")

    lat, errors = [], defaultdict(int)
    t_all = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(one_request, args.url, token,
                            QUESTION_POOL[i % len(QUESTION_POOL)])
                for i in range(args.requests)]
        for i, fut in enumerate(futs):
            sec, code, size = fut.result()
            if code == 200:
                lat.append(sec)
                print(f"[{i+1:>{len(str(args.requests))}}/{args.requests}] "
                      f"{sec:6.2f}s  {size:>6}B  200")
            else:
                errors[code] += 1
                print(f"[{i+1:>{len(str(args.requests))}}/{args.requests}] "
                      f"{sec:6.2f}s  ERR {code} {size}")

    wall = time.perf_counter() - t_all
    lat.sort()
    ok = len(lat)
    print(f"\n{'='*46}")
    print(f"完成 {args.requests} 个请求 | 成功 {ok} | 失败 {sum(errors.values())} "
          f"{dict(errors) if errors else ''}")
    if ok:
        print(f"墙钟 {wall:.1f}s | 吞吐 {ok / wall:.2f} req/s（含真实 LLM 生成）")
        print(f"P50  {percentile(lat, 50):6.2f}s")
        print(f"P95  {percentile(lat, 95):6.2f}s")
        print(f"P99  {percentile(lat, 99):6.2f}s")
        print(f"MIN  {lat[0]:6.2f}s | MAX {lat[-1]:6.2f}s | MEAN "
              f"{sum(lat)/ok:6.2f}s")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
