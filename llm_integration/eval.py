# eval.py

import csv
import json
import time
import requests

URL = "http://localhost:8000/api/chat"
MODEL = "llama3.2:3b"
MAX_TOKENS = 20

SYSTEM_PROMPT = (
    "You are a recycling assistant. Reply in under 10 words using this exact format: "
    "[Recycling, Compost, or Landfill] - [short reason]."
)

def query_llm(item):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Item: {item}"}
        ],
        "stream": True
    }

    start_time = time.time()
    ttft = None
    token_count = 0
    output_text = ""

    response = requests.post(URL, json=payload, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "message" in data and "content" in data["message"]:
                chunk = data["message"]["content"]
                if ttft is None and chunk.strip():
                    ttft = time.time() - start_time

                if "\n" in chunk and token_count > 3:
                    output_text += chunk.split("\n")[0]
                    break

                output_text += chunk
                token_count += 1
                if token_count >= MAX_TOKENS:
                    break

    response.close()
    total_time = time.time() - start_time
    ttft = ttft if ttft is not None else total_time
    gen_time = max(total_time - ttft, 0.001)
    tps = token_count / gen_time

    return output_text.strip(), ttft, tps, total_time

def check_accuracy(output, expected_bin):
    #checks if the expected bin appears before the dash or in the first 25 chars
    prefix = output.split("-")[0].lower() if "-" in output else output[:25].lower()
    return expected_bin.lower() in prefix

def main():
    print("Warming up model on AI HAT+ 2...")
    query_llm("Aluminum Can")
    print("Warmup complete. Starting evaluation...\n")

    results = []
    correct_count = 0

    with open("test_items.csv", mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{'Item':<28} | {'Expected':<10} | {'Pass':<4} | {'TTFT':<6} | {'Tok/s':<5} | {'Total':<6}")
    print("-" * 75)

    for row in rows:
        item = row["item"]
        expected = row["expected_bin"]

        output, ttft, tps, total_time = query_llm(item)
        is_correct = check_accuracy(output, expected)
        if is_correct:
            correct_count += 1

        status = "YES" if is_correct else "NO"
        print(f"{item:<28} | {expected:<10} | {status:<4} | {ttft:>5.2f}s | {tps:>5.1f} | {total_time:>5.2f}s")

        results.append({
            "item": item,
            "expected_bin": expected,
            "correct": is_correct,
            "ttft_sec": round(ttft, 2),
            "tokens_per_sec": round(tps, 2),
            "total_latency_sec": round(total_time, 2),
            "llm_output": output
        })

    #summary stats
    n = len(results)
    accuracy = (correct_count / n) * 100
    avg_ttft = sum(r["ttft_sec"] for r in results) / n
    avg_tps = sum(r["tokens_per_sec"] for r in results) / n
    avg_latency = sum(r["total_latency_sec"] for r in results) / n

    print("\n" + "=" * 45)
    print(f"Bin Accuracy:      {correct_count}/{n} ({accuracy:.1f}%)")
    print(f"Avg TTFT:          {avg_ttft:.2f} s")
    print(f"Avg Throughput:    {avg_tps:.2f} tokens/sec")
    print(f"Avg Total Latency: {avg_latency:.2f} s")
    print("=" * 45)

    with open("eval_results.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved detailed results to eval_results.csv")

if __name__ == "__main__":
    main()
