from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.config import settings, print_startup_info
from app.routers import chat, rag

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="AI Research Assistant - Enhanced",
    description="An intelligent assistant for querying academic literature using modern RAG techniques",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include my enhanced chat router and RAG system
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Enhanced Chat"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG System"])

@app.get("/")
async def root():
    """
    Enhanced home endpoint - shows your system's new capabilities
    """
    return {
        "message": "🧠 AI Research Assistant - Enhanced Version 2.0",
        "status": "ready",
        "version": "2.0.0",
        "creator": "Fatemeh Shahrokhshahi - Enhanced FastAPI + LangChain Integration",
        "improvements_from_v1": [
            "🔧 Professional project structure (app/, services/, routers/)",
            "⚡ LangChain integration for advanced AI capabilities", 
            "💬 Conversation memory and session management",
            "🎯 Enhanced error handling and validation",
            "📊 Performance tracking and monitoring",
            "🏗️ Prepared for RAG system integration"
        ],
        "api_features": [
            "📚 Enhanced chat with conversation history",
            "🔄 Real-time streaming responses", 
            "📋 Session management and conversation tracking",
            "⚙️ Advanced configuration management",
            "🔍 Health monitoring and diagnostics"
        ],
        "next_steps": [
            "Visit /docs for comprehensive API documentation",
            "Try enhanced chat: POST /api/v1/chat/message",
            "View conversations: GET /api/v1/chat/conversations",
            "Check system health: GET /api/v1/chat/health"
        ],
        "coming_soon": [
            "📄 Document upload and processing ✅ READY!",
            "🔍 RAG-powered journal paper queries ✅ READY!",
            "🧮 Vector database integration ✅ READY!",
            "📊 Citation and source tracking ✅ READY!"
        ]
    }

@app.get("/health")
async def system_health():
    """
    Enhanced system health check
    """
    try:
        from app.services.ollama_service import OllamaService
        
        ollama_service = OllamaService()
        ollama_status = await ollama_service.health_check()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "components": {
                "fastapi": "✅ Running smoothly",
                "ollama": ollama_status,
                "langchain": "✅ Integrated",
                "configuration": "✅ Loaded",
                "routers": "✅ Active"
            },
            "performance": ollama_service.get_performance_stats(),
            "configuration": {
                "model": settings.OLLAMA_MODEL,
                "host": f"{settings.HOST}:{settings.PORT}",
                "debug": settings.DEBUG
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "message": "Some system components need attention"
        }

@app.get("/version")
async def version_info():
    """
    Version and upgrade information
    """
    return {
        "current_version": "2.0.0",
        "previous_version": "1.0.0",
        "upgrade_highlights": [
            "Migrated to professional project structure",
            "Added LangChain integration",
            "Enhanced conversation management", 
            "Improved error handling",
            "Added performance monitoring",
            "Prepared RAG system foundation"
        ],
        "architecture_improvements": {
            "configuration": "Centralized settings with environment variables",
            "data_models": "Enhanced Pydantic models with validation",
            "services": "Modular service architecture",
            "routers": "Organized API endpoints",
            "async_support": "Full async/await integration"
        }
    }

@app.get("/test") 
async def legacy_test_endpoint():
    """
    Enhanced version of your original /test endpoint
    """
    try:
        from app.services.ollama_service import OllamaService
        
        ollama_service = OllamaService()
        models = await ollama_service.get_available_models()
        ollama_status = await ollama_service.health_check()
        
        return {
            "message": "🚀 Enhanced system test - All systems upgraded!",
            "fastapi": "✅ Enhanced with professional structure",
            "ollama": ollama_status,
            "langchain": "✅ Integrated and ready",
            "available_models": models,
            "current_model": settings.OLLAMA_MODEL,
            "improvements": [
                "Professional project organization",
                "LangChain integration",
                "Conversation memory",
                "Enhanced error handling",
                "Performance monitoring"
            ],
            "endpoints": {
                "/": "Enhanced home with system info",
                "/api/v1/chat/message": "Enhanced chat with memory",
                "/api/v1/chat/conversations": "Conversation management",
                "/api/v1/chat/health": "Detailed health monitoring",
                "/health": "System health check",
                "/docs": "Comprehensive API documentation"
            }
        }
    except Exception as e:
        return {
            "error": f"System test failed: {str(e)}",
            "suggestion": "Check that Ollama is running: ollama serve"
        }

@app.get("/about")
async def enhanced_about():
    """
    Enhanced version of your original /about endpoint
    """
    return {
        "project": "AI Research Assistant - Enhanced Edition",
        "creator": "Fatemeh Shahrokhshahi",
        "education": "Master of Computer Engineering, Istanbul Aydin University",
        "research": "Specializing in LLM reasoning, domain clustering, and RAG systems",
        "github": "github.com/fatemeshahrokhshahi",
        "version_history": {
            "v1.0": "Basic FastAPI chatbot with Ollama",
            "v2.0": "Enhanced with LangChain, conversation memory, and RAG preparation"
        },
        "technologies_v2": [
            "FastAPI - Modern async web framework",
            "LangChain - Advanced AI application framework",
            "Ollama - Local AI model integration",
            "Pydantic - Data validation and settings",
            "ChromaDB - Vector database (coming soon)",
            "Sentence Transformers - Embeddings (coming soon)"
        ],
        "learning_achievements": [
            "Professional Python project structure",
            "Advanced FastAPI patterns",
            "LangChain integration",
            "Async programming mastery",
            "Configuration management",
            "API design and documentation"
        ],
        "coming_next": [
            "RAG system for academic papers",
            "Document processing pipeline", 
            "Vector database integration",
            "Citation tracking and source attribution"
        ]
    }

if __name__ == "__main__":
    print_startup_info()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )