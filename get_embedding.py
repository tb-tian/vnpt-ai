import requests
import json

def get_embedding(text):
    # Load API keys from api-keys.json
    with open('api-keys.json', 'r') as f:
        api_keys = json.load(f)

    # Find the embedding API key
    embedding_key = next((item for item in api_keys if item["llmApiName"] == "LLM embedings"), None)

    if not embedding_key:
        raise ValueError("Embedding API key not found in api-keys.json")

    headers = {
        'Authorization': embedding_key['authorization'],
        'Token-id': embedding_key['tokenId'],
        'Token-key': embedding_key['tokenKey'],
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': 'vnptai_hackathon_embedding',
        'input': text,
        'encoding_format': 'float'
    }

    response = requests.post(
        'https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding',
        headers=headers,
        json=json_data
    )

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    return response.json()['data'][0]['embedding']

if __name__ == "__main__":
    text = 'Xin chào, mình là VNPT ΑΙ.'
    embedding = get_embedding(text)
    print(embedding)