import requests

response = requests.get("http://localhost:8000/")
print("Status Code:", response.status_code)
print("Response Body:", response.text)