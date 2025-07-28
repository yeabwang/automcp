import requests

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": "Bearer REDACTED_GROQ_KEY",
    "Content-Type": "application/json"
}
payload = {
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "messages": [
        {"role": "user", "content": "Hello, what is the weather today?"}
    ],
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=payload)
print(response.status_code)
print(response.json())