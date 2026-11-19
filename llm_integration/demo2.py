
import requests
import json
import time

def run_tech_demo(classification_input):
    url = "http://localhost:8000/api/chat"
    
    system_prompt = (
        "You are a recycling assistant. Reply in ONE short sentence only. "
        "Format: [Bin: Recycling, Compost, or Landfill] - [brief reason]."
    )
    
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Item: {classification_input}"}
        ],
        "stream": True
    }
    
    print(f"Mock Vision CNN Input: '{classification_input}'\n")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        print("Output: ", end="", flush=True)
        token_count = 0
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if "message" in data and "content" in data["message"]:
                    chunk = data["message"]["content"]
                    
                    # Stop immediately at the first newline or after 30 tokens
                    if "\n" in chunk and token_count > 5:
                        print(chunk.split("\n")[0], end="", flush=True)
                        break
                        
                    print(chunk, end="", flush=True)
                    token_count += 1
                    
                    if token_count >= 30:
                        break
                        
        response.close()
        print("\n")
        
    except requests.exceptions.RequestException as e:
        print(f"\nConnection Error: {e}")
        
    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_tech_demo("PET Plastic Water Bottle")
