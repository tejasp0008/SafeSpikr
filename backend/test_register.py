import requests, sys
url = "http://localhost:8000/register_user"
fp = r"C:\Users\TEJAS PAL\OneDrive\Desktop\tejas_photo.jpg"
files = [("frames", open(fp, "rb"))]
data = {"name": "Tejas"}
try:
    r = requests.post(url, files=files, data=data, timeout=20)
    print("STATUS:", r.status_code)
    print("HEADERS:", r.headers)
    print("BODY:", r.text)
except Exception as e:
    print("REQUEST ERROR:", e)
    sys.exit(1)
