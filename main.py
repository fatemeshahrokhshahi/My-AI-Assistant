from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import List, Optional

app = FastAPI(title="My AI Assistant - Local & Free", version="1.0.0")

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "tinyllama"

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    model_used: str
    status: str

def query_ollama(prompt: str, model: str = DEFAULT_MODEL):
    """Query local Ollama instance"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30  # 30 seconds should be enough for most questions
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("response", "No response generated")
            return ai_response
        else:
            return "Ollama server error. Make sure 'ollama serve' is running."
            
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama not running. Please run 'ollama serve' in another terminal."
    except requests.exceptions.Timeout:
        return "⏱️ Question too complex. Try asking something simpler or wait a bit longer."
    except Exception as e:
        return f"Local AI error: {str(e)}"

@app.get("/")
def home():
    return {
        "message": "🤖 My AI Assistant - Local & Free!", 
        "status": "ready",
        "model": DEFAULT_MODEL,
        "features": [
            "Local AI running on your computer",
            "100% free with no API costs",
            "Works offline",
            "No data sent to external servers"
        ],
        "creator": "Built by Fatemeh Shahrokhshahi",
        "next": "Visit /docs to try the chat feature!"
    }

@app.post("/chat", response_model=ChatResponse)
def chat_with_ai(chat: ChatMessage):
    """Chat with your local AI assistant"""
    try:
        ai_response = query_ollama(chat.message)
        
        return ChatResponse(
            response=ai_response,
            model_used=f"ollama/{DEFAULT_MODEL}",
            status="success"
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Sorry, I had an error: {str(e)}",
            model_used="error",
            status="error"
        )

@app.get("/test")
def test():
    """Test all systems"""
    # Test Ollama connection
    try:
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code == 200:
            models = test_response.json().get("models", [])
            ollama_status = f"✅ Ollama running with {len(models)} model(s)"
            available_models = [model["name"] for model in models]
        else:
            ollama_status = "⚠️ Ollama server issue"
            available_models = []
    except:
        ollama_status = "❌ Ollama not running - run 'ollama serve'"
        available_models = []
    
    return {
        "fastapi": "✅ Working perfectly",
        "ollama": ollama_status,
        "available_models": available_models,
        "current_model": DEFAULT_MODEL,
        "endpoints": {
            "/": "Home page",
            "/chat": "Chat with AI",
            "/test": "System status",
            "/models": "Available models", 
            "/docs": "API documentation"
        }
    }

@app.get("/models")
def available_models():
    """List and manage Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {
                "available_models": [model["name"] for model in models],
                "current_default": DEFAULT_MODEL,
                "how_to_install": "Use 'ollama pull model_name' to install new models",
                "recommended_models": {
                    "phi": "✅ Currently using - Fast and capable (2.7B)",
                    "llama2": "More capable but slower (7B)", 
                    "mistral": "Good balance of speed/capability (7B)",
                    "codellama": "Specialized for programming (7B)",
                    "tinyllama": "Very fast, smaller responses (1.1B)"
                },
                "note": "Larger models give better answers but are slower"
            }
    except:
        return {"error": "Ollama not running - start with 'ollama serve'"}

@app.get("/about")
def about():
    """About this project"""
    return {
        "project": "AI Assistant - Learning FastAPI + Local AI",
        "creator": "Fatemeh Shahrokhshahi",
        "education": "Master of Computer Engineering, Istanbul Aydin University", 
        "research": "Specializing in LLM reasoning and domain clustering",
        "github": "github.com/fatemeshahrokhshahi",
        "technologies": [
            "FastAPI - Modern Python web framework",
            "Ollama - Local AI model runner", 
            "Phi model - Microsoft's efficient language model"
        ],
        "learning_goals": [
            "Master FastAPI development",
            "Integrate AI models into applications",
            "Build production-ready APIs",
            "Prepare for AI engineering roles"
        ]
    }