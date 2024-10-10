import os
import threading
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Union
from cerebras.cloud.sdk import Cerebras
from groq import Groq
import openai
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB  # For PostgreSQL JSONB field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Thread lock for synchronizing access to api_keys.json
api_keys_lock = threading.Lock()

# Database setup
# DATABASE_URL = os.environ.get("DATABASE_URL")  # Ensure this is set in your environment

# if not DATABASE_URL:
#     raise Exception("DATABASE_URL environment variable not set")

# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# Define the ChatLog model
# class ChatLog(Base):
#     __tablename__ = 'chat_logs'

#     id = Column(Integer, primary_key=True, index=True)
#     timestamp = Column(DateTime(timezone=True), server_default=func.now())
#     client_name = Column(String(100))
#     client_email = Column(String(100))
#     service_name = Column(String(50))
#     model_name = Column(String(100))
#     client_message = Column(Text)
#     content_in_response = Column(Text)
#     raw_response = Column(JSONB)

# # Create the tables in the database
# Base.metadata.create_all(bind=engine)

# Master API keys
MASTER_SERVICE_API_KEYS = {
    # "cerebras": os.environ.get("CEREBRAS_API_KEY"),
    # "groq": os.environ.get("GROQ_API_KEY"),
    # "sambanova": os.environ.get("SAMBANOVA_API_KEY"),
    # Add more services as needed
    "cerebras": "csk-e2e8kypw838rwmpjxd9nx2vn5jrertm339fnrcnt9c6p8hmx",
    "groq": "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",
    "sambanova": "2e1e850b-24f9-4abd-89a6-5cfc61d4ca50"
}

for service, api_key in MASTER_SERVICE_API_KEYS.items():
    if not api_key:
        raise Exception(f"Master API key for {service} is not set in environment variables.")
    
# Admin API key for sensitive operations
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    raise Exception("ADMIN_API_KEY environment variable not set")
    
# Available models for each service
SERVICE_MODELS = {
    "groq": {
        "gemma-7b-it",
        "gemma-9b-it",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "distil-whisper-large-v3-en",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-11b-text-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-90b-text-preview",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
        "llava-v1.5-7b-4096-preview"
    },
    "cerebras": {
        "llama3.1-8b",
        "llama3.1-70b"
    },
    "sambanova": {
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
        "Meta-Llama-3.2-1B-Instruct",
        "Meta-Llama-3.2-3B-Instruct"
    }
}

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    service_name: str
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1024, ge=1)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

    @validator('messages')
    def messages_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('The "messages" field must contain at least one message.')
        return v

    @validator('temperature')
    def temperature_in_range(cls, v):
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError('The "temperature" must be between 0.0 and 2.0.')
        return v

    @validator('max_tokens')
    def max_tokens_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('The "max_tokens" must be a positive integer.')
        return v

    @validator('top_p')
    def top_p_in_range(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError('The "top_p" must be between 0.0 and 1.0.')
        return v

    @validator('stream')
    def stream_must_be_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError('The "stream" parameter must be a boolean.')
        return v

    @validator('stop')
    def stop_valid(cls, v):
        if v is not None:
            if isinstance(v, str):
                if not v.strip():
                    raise ValueError('The "stop" string must not be empty.')
            elif isinstance(v, list):
                if not all(isinstance(item, str) and item.strip() for item in v):
                    raise ValueError('All elements in "stop" list must be non-empty strings.')
            else:
                raise ValueError('The "stop" parameter must be a string or a list of strings.')
        return v

class ChatCompletionResponse(BaseModel):
    content: str

class GenerateApiKeyRequest(BaseModel):
    name: str
    email: EmailStr
    privilege: str = 'user'
    
class DeleteApiKeyRequest(BaseModel):
    email: EmailStr

def generate_api_key(length=30):
    import random
    import string
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def initialize_api_keys_file():
    """Initializes the api_keys.json file if it doesn't exist or is invalid."""
    with api_keys_lock:
        if not os.path.exists('api_keys.json') or os.stat('api_keys.json').st_size == 0:
            with open('api_keys.json', 'w') as f:
                json.dump({}, f, indent=4)
                logger.info("Initialized api_keys.json with an empty dictionary.")
        else:
            # Check if the file contains valid JSON
            with open('api_keys.json', 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError:
                    logger.warning("api_keys.json contains invalid JSON. Re-initializing the file.")
                    with open('api_keys.json', 'w') as f:
                        json.dump({}, f, indent=4)

def load_api_keys():
    """Loads API keys from the JSON file."""
    with api_keys_lock:
        if not os.path.exists('api_keys.json'):
            return {}
        with open('api_keys.json', 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning("api_keys.json is empty or contains invalid JSON. Returning empty dictionary.")
                return {}

def save_api_keys(api_keys):
    """Saves API keys to the JSON file."""
    with api_keys_lock:
        with open('api_keys.json', 'w') as f:
            json.dump(api_keys, f, indent=4)

def add_api_key(client_name, client_email, privilege='user'):
    """Generates a new API key, adds it to api_keys.json, and returns the key."""
    api_keys = load_api_keys()
    # Check if email already exists
    for key, info in api_keys.items():
        if info['email'].lower() == client_email.lower():
            raise ValueError("An API key has already been generated for this email.")
    new_key = generate_api_key()
    api_keys[new_key] = {
        "name": client_name,
        "email": client_email,
        "privilege": privilege
    }
    save_api_keys(api_keys)
    return new_key

def authenticate_client(api_key: str = Header(..., alias="api-key")):
    """Dependency function to authenticate the client and return their info."""
    api_keys = load_api_keys()
    if api_key not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_keys[api_key]


def authenticate_client_optional(api_key: str = Header(None, alias="api-key")):
    """Dependency function to optionally authenticate the client."""
    if api_key is None:
        return None
    return authenticate_client(api_key)

def authenticate_admin(admin_api_key: str = Header(..., alias="admin-api-key")):
    """Dependency function to authenticate admin for sensitive operations."""
    if admin_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid Admin API Key")
    return admin_api_key

def get_master_client(service_name: str):
    """Returns the appropriate master API client based on the service name."""
    service_name = service_name.lower()
    if service_name == "cerebras":
        api_key = MASTER_SERVICE_API_KEYS["cerebras"]
        return Cerebras(api_key=api_key)
    elif service_name == "groq":
        api_key = MASTER_SERVICE_API_KEYS["groq"]
        return Groq(api_key=api_key)
    elif service_name == "sambanova":
        # No client needed; handled in the endpoint
        return None
    else:
        raise ValueError(f"Unsupported service_name: {service_name}")

def log_to_database(log_entry):
    """Logs the data into the database."""
    session = SessionLocal()
    try:
        session.add(log_entry)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error logging to database: {e}")
    finally:
        session.close()
        
def serialize_chat_completion(chat_completion):
    """Serializes the chat_completion object to a JSON-serializable format."""
    try:
        return chat_completion.dict()
    except AttributeError:
        pass

    try:
        return vars(chat_completion)
    except TypeError:
        pass

    # Custom serialization
    import json

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)

    return json.loads(json.dumps(chat_completion, cls=CustomEncoder))

# Initialize the api_keys.json file at startup
initialize_api_keys_file()

@app.post("/generate_api_key")
async def generate_api_key_endpoint(
    request: GenerateApiKeyRequest,
    api_key_info: dict = Depends(authenticate_client_optional),
):
    """API endpoint to generate a new API key."""
    client_name = request.name
    client_email = request.email
    privilege = request.privilege.lower()
    
    if privilege not in ['user', 'admin']:
        raise HTTPException(status_code=400, detail="Invalid privilege level")
    
    if privilege == 'admin':
        # Require authentication as admin
        if api_key_info is None or api_key_info.get('privilege') != 'admin':
            raise HTTPException(status_code=403, detail="Admin privilege required to generate admin API key")
    
    try:
        new_key = add_api_key(client_name, client_email, privilege)
        return {"api_key": new_key}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

@app.delete("/delete_api_key")
async def delete_api_key_endpoint(
    request: DeleteApiKeyRequest,
    api_key_info: dict = Depends(authenticate_client),
):
    """API endpoint to delete an API key by email."""
    if api_key_info.get('privilege') != 'admin':
        raise HTTPException(status_code=403, detail="Admin privilege required to delete API keys")
    
    client_email = request.email.lower()
    
    with api_keys_lock:
        api_keys = load_api_keys()
        key_to_delete = None
        for key, info in api_keys.items():
            if info['email'].lower() == client_email:
                key_to_delete = key
                break
    
        if key_to_delete:
            del api_keys[key_to_delete]
            save_api_keys(api_keys)
            return {"detail": f"API key associated with email {client_email} has been deleted."}
        else:
            raise HTTPException(status_code=404, detail=f"No API key found for email {client_email}.")

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(authenticate_client),
):
    service_name = request.service_name.lower()
    model_name = request.model

    try:
        # Validate the service name
        if service_name not in SERVICE_MODELS:
            raise ValueError(f"Unsupported service_name: {service_name}")

        # Validate the model availability
        available_models = SERVICE_MODELS[service_name]
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' is not available for service '{service_name}'.")

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

        # Prepare parameters for the API call
        api_call_params = {
            "messages": messages,
            "model": model_name,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": request.stream,
            "stop": request.stop,
        }

        # Remove parameters with None values to avoid issues
        api_call_params = {k: v for k, v in api_call_params.items() if v is not None}

        # Process the request and capture the raw response
        if service_name == "cerebras":
            chat_completion = client.chat.completions.create(**api_call_params)
            content_in_response = chat_completion.choices[0].message.content

        elif service_name == "groq":
            chat_completion = client.chat.completions.create(**api_call_params)
            content_in_response = chat_completion.choices[0].message.content
        
        elif service_name == "sambanova":
            # Add API key and base URL to parameters
            api_call_params["api_key"] = MASTER_SERVICE_API_KEYS['sambanova']
            api_call_params["api_base"] = "https://api.sambanova.ai/v1"

            # Process the request and capture the raw response
            chat_completion = openai.ChatCompletion.create(**api_call_params)
            content_in_response = chat_completion.choices[0].message.content

        else:
            raise ValueError(f"Unsupported service_name: {service_name}")

        # Serialize the raw response
        raw_response = serialize_chat_completion(chat_completion)

        # Log the data to the database
        # log_entry = ChatLog(
        #     client_name=client_name,
        #     client_email=client_email,
        #     service_name=service_name,
        #     model_name=model_name,
        #     client_message=client_message.strip(),
        #     content_in_response=content_in_response.strip(),
        #     raw_response=raw_response  # New field
        # )
        # log_to_database(log_entry)

        return ChatCompletionResponse(content=content_in_response)

    except ValueError as ve:
        logger.error(f"ValueError in chat_completions endpoint: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in chat_completions endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")