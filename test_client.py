import requests

# Change to your EC2 IP
BASE_URL = "http://YOUR_EC2_PUBLIC_IP:8000"

# Test phrases - Chennai slang
phrases = [
    "Vanakkam da, enna da panra?",
    "Dei machan, sema mass da scene-u!",
    "Seri da, naan office mudichitu varen",
    "Chennai la mazhai sema ya irukku da",
    "Saptiya machan? Illa da, innum illa"
]

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("Health:", r.json())

def test_synthesize():
    for text in phrases:
        print(f"\nSynthesizing: {text}")
        resp = requests.post(f"{BASE_URL}/synthesize", json={"text": text, "cache": True})
        if resp.status_code == 200:
            filename = f"output_{text[:10].replace(' ', '_')}.wav"
            with open(filename, "wb") as f:
                f.write(resp.content)
            print(f"Saved -> {filename} ({len(resp.content)} bytes)")
        else:
            print(f"Failed: {resp.status_code} {resp.text}")

if __name__ == "__main__":
    test_health()
    test_synthesize()
