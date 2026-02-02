import requests
import sys

try:
    print("Testing connection to http://127.0.0.1:8000/docs ...")
    response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
    if response.status_code == 200:
        print("SUCCESS: Backend is reachable!")
    else:
        print(f"WARNING: Backend returned status code {response.status_code}")
except Exception as e:
    print(f"ERROR: Could not connect to backend: {e}")
    print("Make sure python web_api.py is running.")
