import requests
import json

def get_response(messages, model="small", temperature=1.0, max_tokens=1000, 
                 n=1, logprobs=None, top_logprobs=None, response_format=None, **kwargs):
    """
    Get response from VNPT AI API
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: "small" or "large" (default: "small")
        temperature: Temperature for sampling (default: 1.0)
        max_tokens: Max completion tokens (default: 1000)
        n: Number of completions to generate (default: 1)
        logprobs: Whether to return log probabilities (default: None)
        top_logprobs: Number of most likely tokens to return (default: None)
        response_format: Output format, e.g., {"type": "json_object"} (default: None)
        **kwargs: Additional API parameters
    
    Returns:
        str or dict: Response content from the model (or full response if n > 1)
    """
    # Load API keys from api-keys.json
    with open('api-keys.json', 'r') as f:
        api_keys = json.load(f)

    # Select API key based on model
    if model == "large":
        model_name = "LLM large"
        api_model = "vnptai_hackathon_large"
        api_endpoint = "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large"
    else:  # default to small
        model_name = "LLM small"
        api_model = "vnptai_hackathon_small"
        api_endpoint = "https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small"

    chat_key = next((item for item in api_keys if item["llmApiName"] == model_name), None)

    if not chat_key:
        raise ValueError(f"{model_name} API key not found in api-keys.json")

    headers = {
        'Authorization': chat_key['authorization'],
        'Token-id': chat_key['tokenId'],
        'Token-key': chat_key['tokenKey'],
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': api_model,
        'messages': messages,
        'temperature': temperature,
        'top_p': 1.0,
        'top_k': 20,
        'n': n,
        'max_completion_tokens': max_tokens,
    }
    
    # Add optional parameters if provided
    if logprobs is not None:
        json_data['logprobs'] = logprobs
    if top_logprobs is not None:
        json_data['top_logprobs'] = top_logprobs
    if response_format is not None:
        json_data['response_format'] = response_format
    
    # Add any additional kwargs
    json_data.update(kwargs)

    response = requests.post(api_endpoint, headers=headers, json=json_data)

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    response_data = response.json()
    
    # If n > 1 or logprobs requested, return full response for analysis
    if n > 1 or logprobs is not None:
        return response_data
    
    # Otherwise return just the content string (backward compatible)
    return response_data['choices'][0]['message']['content']

if __name__ == "__main__":
    messages = [
        {
            'role': 'user',
            'content': 'Hi, VNPT AI.',
        },
    ]
    response_text = get_response(messages)
    print(response_text)