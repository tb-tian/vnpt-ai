import requests
import json

def get_response(messages):
    # Load API keys from api-keys.json
    with open('api-keys.json', 'r') as f:
        api_keys = json.load(f)

    # Find the chat API key
    chat_key = next((item for item in api_keys if item["llmApiName"] == "LLM small"), None)

    if not chat_key:
        raise ValueError("Chat API key not found in api-keys.json")

    headers = {
        'Authorization': chat_key['authorization'],
        'Token-id': chat_key['tokenId'],
        'Token-key': chat_key['tokenKey'],
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': 'vnptai_hackathon_small',
        'messages': messages,
        'temperature': 1.0,
        'top_p': 1.0,
        'top_k': 20,
        'n': 1,
        'max_completion_tokens': 1000,
    }

    response = requests.post(
        'https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small',
        headers=headers,
        json=json_data
    )

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    return response.json()['choices'][0]['message']['content']

if __name__ == "__main__":
    messages = [
        {
            'role': 'user',
            'content': 'Hi, VNPT AI.',
        },
    ]
    response_text = get_response(messages)
    print(response_text)