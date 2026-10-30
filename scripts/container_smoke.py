import time

import requests

from scripts.deploy import event

url = "http://localhost:9000/2015-03-31/functions/function/invocations"
for _i in range(30):
    try:
        response = requests.post(url, json=event("/health"), timeout=10)
        if response.ok and response.json().get("statusCode") == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError("Lambda container health failed")
response = requests.post(url, json=event("/search", "POST", body={"question": "test"}), timeout=10)
if response.json().get("statusCode") != 401:
    raise RuntimeError("Unauthenticated container request was not blocked")
print("Lambda runtime health/authentication smoke passed")
