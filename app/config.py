# app/config.py - Enhanced configuration for your AI assistant

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Enhanced settings for your AI Research Assistant.
    
    This replaces the simple constants in your original main.py
    with a professional configuration system.
    """
    
    # === API Configuration ===
    APP_NAME: str = "AI Research Assistant - Enhanced"
    VERSION: str = "2.0.0"
    DEBUG: bool = True  # Set to False in production
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # === Your Original Ollama Configuration (Enhanced) ===
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "tinyllama"  # Your current default model
    OLLAMA_TIMEOUT: int = 30
    
    # === New RAG System Configuration ===
    # Vector Database Settings
    VECTOR_DB_TYPE: str = "chromadb"  # We'll use ChromaDB for simplicity
    VECTOR_DB_PATH: str = "./data/vectorstore"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Document Processing Settings (for your journal papers)
    CHUNK_SIZE: int = 1000  # Characters per document chunk
    CHUNK_OVERLAP: int = 200  # Overlap between chunks for better context
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB max file size
    
    # RAG Retrieval Settings
    RETRIEVAL_K: int = 5  # Number of relevant chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.7  # Minimum similarity score
    
    # === File Storage Configuration ===
    UPLOAD_DIR: str = "./data/uploads"
    PROCESSED_DIR: str = "./data/processed"
    DOCUMENTS_DIR: str = "./data/documents"
    
    # === Performance Settings ===
    MAX_WORKERS: int = 4  # For async processing
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create a global settings instance
settings = Settings()

# === Helper Functions ===

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.UPLOAD_DIR,
        settings.PROCESSED_DIR,
        settings.VECTOR_DB_PATH,
        settings.DOCUMENTS_DIR,
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def print_startup_info():
    """Print startup information (replaces your original startup messages)"""
    print("🚀 Starting AI Research Assistant - Enhanced Version")
    print(f"📍 Host: {settings.HOST}:{settings.PORT}")
    print(f"🤖 Ollama: {settings.OLLAMA_URL} (model: {settings.OLLAMA_MODEL})")
    print(f"🗄️ Vector DB: {settings.VECTOR_DB_TYPE} at {settings.VECTOR_DB_PATH}")
    print(f"📚 Ready for journal paper processing and RAG queries!")

# Initialize directories when module is imported
ensure_directories()