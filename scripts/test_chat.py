"""Test the chat endpoint."""

import urllib.request
import urllib.error
import json

BASE_URL = "http://localhost:3000/api"

def test_health():
    """Test the health endpoint."""
    print("Testing /api/health endpoint...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=10) as response:
            print(f"Status: {response.status}")
            print(f"Response: {response.read().decode()}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        print(f"Response: {e.read().decode()}")
    except Exception as e:
        print(f"Error: {e}")
    print("---")

def test_chat():
    """Test the chat endpoint."""
    print("Testing /api/chat endpoint...")
    try:
        data = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": "Say hello in one word"
                }
            ]
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{BASE_URL}/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            print(f"Status: {response.status}")
            print(f"Headers: {dict(response.headers)}")
            body = response.read().decode()
            print(f"Response (first 2000 chars): {body[:2000]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        body = e.read().decode()
        print(f"Response: {body[:2000]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_health()
    test_chat()
