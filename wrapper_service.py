import os
import threading
import json
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from cerebras.cloud.sdk import Cerebras
from groq import Groq

app = FastAPI()

# Thread lock for synchronizing access to api_keys.json
api_keys_lock = threading.Lock()

# API keys for master services (stored securely)
MASTER_SERVICE_API_KEYS = {
    "cerebras": os.environ.get("CEREBRAS_API_KEY"),
    "groq": os.environ.get("GROQ_API_KEY"),
    # Add more services as needed
}

# Ensure all necessary master API keys are set
for service, api_key in MASTER_SERVICE_API_KEYS.items():
    if not api_key:
        raise Exception(f"Master API key for {service} is not set in environment variables.")

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    service_name: str  # Service to use: 'cerebras' or 'groq'

class ChatCompletionResponse(BaseModel):
    content: str

class GenerateApiKeyRequest(BaseModel):
    name: str
    email: EmailStr

def generate_api_key(length=30):
    """Generates a random API key of the given length."""
    import random
    import string
    characters = string.ascii_letters + string.digits
    api_key = ''.join(random.choice(characters) for _ in range(length))
    return api_key

def load_api_keys():
    """Loads API keys from the JSON file."""
    with api_keys_lock:
        if not os.path.exists('api_keys.json'):
            return {}
        with open('api_keys.json', 'r') as f:
            return json.load(f)

def save_api_keys(api_keys):
    """Saves API keys to the JSON file."""
    with api_keys_lock:
        with open('api_keys.json', 'w') as f:
            json.dump(api_keys, f, indent=4)

def add_api_key(client_name, client_email):
    """Generates a new API key, adds it to api_keys.json, and returns the key."""
    api_keys = load_api_keys()
    new_key = generate_api_key()
    api_keys[new_key] = {
        "name": client_name,
        "email": client_email
    }
    save_api_keys(api_keys)
    return new_key

def authenticate_client(api_key: str = Header(...)):
    """Dependency function to authenticate the client."""
    api_keys = load_api_keys()
    if api_key not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key  # Return the API key if needed

def get_master_client(service_name: str):
    """Returns the appropriate master API client based on the service name."""
    service_name = service_name.lower()
    if service_name == "cerebras":
        api_key = MASTER_SERVICE_API_KEYS["cerebras"]
        return Cerebras(api_key=api_key)
    elif service_name == "groq":
        api_key = MASTER_SERVICE_API_KEYS["groq"]
        return Groq(api_key=api_key)
    else:
        raise ValueError(f"Unsupported service_name: {service_name}")

# Endpoint to generate a new API key for the client
@app.post("/generate_api_key")
async def generate_api_key_endpoint(request: GenerateApiKeyRequest):
    """API endpoint to generate a new API key."""
    client_name = request.name
    client_email = request.email

    # Optionally, check if the email already exists
    api_keys = load_api_keys()
    for key_info in api_keys.values():
        if key_info['email'].lower() == client_email.lower():
            raise HTTPException(status_code=400, detail="An API key has already been generated for this email.")

    new_key = add_api_key(client_name, client_email)
    return {"api_key": new_key}

# Endpoint to handle chat completions
@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(authenticate_client),
):
    service_name = request.service_name.lower()

    try:
        client = get_master_client(service_name)
        messages = [message.dict() for message in request.messages]

        if service_name == "cerebras":
            # Cerebras API call
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=request.model,
            )
            # Adjust based on actual response from Cerebras
            content = chat_completion.get('content', '')

        elif service_name == "groq":
            # Groq API call
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=request.model,
            )
            # Extract the response content
            content = chat_completion.choices[0].message.content

        else:
            raise ValueError(f"Unsupported service_name: {service_name}")

        # Return the response to the client
        return ChatCompletionResponse(content=content)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))