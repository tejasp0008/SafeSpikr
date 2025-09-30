import requests

url = "http://127.0.0.1:8000/predict"
with open("frame.jpg", "rb") as f:
    files = {"frame": ("frame.jpg", f, "image/jpeg")}
    data = {"user_id": "1"}
    resp = requests.post(url, files=files, data=data)
    print("STATUS:", resp.status_code)
    print(resp.text)
