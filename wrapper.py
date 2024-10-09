import os
import threading
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from cerebras.cloud.sdk import Cerebras
from groq import Groq
import openpyxl
from openpyxl import Workbook

app = FastAPI()

# Thread locks
api_keys_lock = threading.Lock()
excel_lock = threading.Lock()

# Paths
EXCEL_FILE = 'chat_logs.xlsx'

# Initialize the Excel file
def initialize_excel_file():
    """Initializes the Excel file with headers if it doesn't exist."""
    if not os.path.exists(EXCEL_FILE):
        with excel_lock:
            wb = Workbook()
            ws = wb.active
            ws.title = "Chat Logs"
            headers = [
                "Date and Time",
                "Client Name",
                "Client Email",
                "Service Name",
                "Model Name",
                "Client Message",
                "Content in Response"
            ]
            ws.append(headers)
            wb.save(EXCEL_FILE)

initialize_excel_file()

# Master API keys
MASTER_SERVICE_API_KEYS = {
    "cerebras": os.environ.get("CEREBRAS_API_KEY"),
    "groq": os.environ.get("GROQ_API_KEY"),
    # Add more services as needed
}

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
    service_name: str

class ChatCompletionResponse(BaseModel):
    content: str

class GenerateApiKeyRequest(BaseModel):
    name: str
    email: EmailStr

def generate_api_key(length=30):
    import random
    import string
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def load_api_keys():
    with api_keys_lock:
        if not os.path.exists('api_keys.json'):
            return {}
        with open('api_keys.json', 'r') as f:
            return json.load(f)

def save_api_keys(api_keys):
    with api_keys_lock:
        with open('api_keys.json', 'w') as f:
            json.dump(api_keys, f, indent=4)

def add_api_key(client_name, client_email):
    api_keys = load_api_keys()
    new_key = generate_api_key()
    api_keys[new_key] = {
        "name": client_name,
        "email": client_email
    }
    save_api_keys(api_keys)
    return new_key

def authenticate_client(api_key: str = Header(...)):
    api_keys = load_api_keys()
    if api_key not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

def get_master_client(service_name: str):
    service_name = service_name.lower()
    if service_name == "cerebras":
        api_key = MASTER_SERVICE_API_KEYS["cerebras"]
        return Cerebras(api_key=api_key)
    elif service_name == "groq":
        api_key = MASTER_SERVICE_API_KEYS["groq"]
        return Groq(api_key=api_key)
    else:
        raise ValueError(f"Unsupported service_name: {service_name}")

def log_to_excel(log_data):
    with excel_lock:
        wb = openpyxl.load_workbook(EXCEL_FILE)
        ws = wb.active
        ws.append(log_data)
        wb.save(EXCEL_FILE)

@app.post("/generate_api_key")
async def generate_api_key_endpoint(request: GenerateApiKeyRequest):
    client_name = request.name
    client_email = request.email

    api_keys = load_api_keys()
    for key_info in api_keys.values():
        if key_info['email'].lower() == client_email.lower():
            raise HTTPException(status_code=400, detail="An API key has already been generated for this email.")

    new_key = add_api_key(client_name, client_email)
    return {"api_key": new_key}

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(authenticate_client),
):
    service_name = request.service_name.lower()
    model_name = request.model

    try:
        client = get_master_client(service_name)
        messages = [message.dict() for message in request.messages]

        # Retrieve client information
        api_keys = load_api_keys()
        client_info = api_keys[api_key]
        client_name = client_info['name']
        client_email = client_info['email']

        # Extract client message
        client_message = ' '.join(
            [msg.content for msg in request.messages if msg.role == 'user']
        )

        # Process the request
        if service_name == "cerebras":
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
            )
            content_in_response = chat_completion.get('content', '')

        elif service_name == "groq":
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
            )
            content_in_response = chat_completion.choices[0].message.content

        else:
            raise ValueError(f"Unsupported service_name: {service_name}")

        # Log the data
        log_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            client_name,
            client_email,
            service_name,
            model_name,
            client_message.strip(),
            content_in_response.strip()
        ]
        log_to_excel(log_data)

        return ChatCompletionResponse(content=content_in_response)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))