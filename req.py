import requests

url = "http://localhost:8000/query"
payload = {
    "system": "你是一个代码reviewer。",
    "user_input": "请review 代码：print('hello world')",
    "max_new_tokens": 150
}

response = requests.post(url, json=payload)
print(response.json())