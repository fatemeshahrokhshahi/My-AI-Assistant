# app/routers/chat.py - Enhanced chat router with conversation management

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Optional
import uuid
import asyncio
import json
from datetime import datetime

from app.models.chat_models import (
    ChatMessage, ChatResponse, Conversation, 
    ConversationMessage, MessageRole
)
# Fixed imports - add LangChainService
from app.services.ollama_service import OllamaService
from app.services.langchain_service import LangChainService
from app.config import settings

# Create router
router = APIRouter()

# Global conversation storage (in production, use Redis or database)
# We'll improve this in later phases
conversations: Dict[str, Conversation] = {}

# Initialize services - both services now
ollama_service = OllamaService()
langchain_service = LangChainService()

@router.post("/message", response_model=ChatResponse)
async def chat_message(message: ChatMessage, background_tasks: BackgroundTasks):
    """
    Send a message to the AI assistant.
    
    This endpoint handles:
    - New conversations and existing conversation continuity
    - Optional RAG integration (we'll implement this in Phase 3)
    - Conversation history management
    - Performance tracking
    """
    
    try:
        # Generate session ID if not provided
        session_id = message.session_id or str(uuid.uuid4())
        
        # Get or create conversation
        if session_id not in conversations:
            conversations[session_id] = Conversation(session_id=session_id)
        
        conversation = conversations[session_id]
        
        # Add user message to conversation history
        conversation.add_message(
            role=MessageRole.USER,
            content=message.message,
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # Prepare conversation context for AI
        conversation_context = conversation.get_context(max_messages=10)
        
        # TODO: In Phase 3, we'll add RAG retrieval here
        context = None
        if message.use_rag:
            # Placeholder for RAG integration
            context = "RAG system not yet implemented - will be added in Phase 3"
        
        # Enhanced AI response using LangChain
        if message.use_rag or len(conversation.messages) > 4:  # Use LangChain for complex conversations
            ai_response = await langchain_service.chat_with_memory(
                message=message.message,
                session_id=session_id,
                conversation_history=conversation.messages[:-1],  # Exclude the just-added user message
                use_summary_memory=(len(conversation.messages) > 20)
            )
        else:
            # Use basic Ollama service for simple queries
            ai_response = await ollama_service.chat_completion(
                messages=conversation_context,
                context=context,
                use_rag=message.use_rag,
                temperature=0.1
            )
        
        # Add AI response to conversation history
        conversation.add_message(
            role=MessageRole.ASSISTANT,
            content=ai_response["response"],
            metadata={
                "model_used": ai_response["model_used"],
                "response_time": ai_response["response_time"],
                "context_used": ai_response.get("context_used", False)
            }
        )
        
        # Generate conversation title if this is the first exchange (using LangChain)
        if len(conversation.messages) == 2 and not conversation.title:
            background_tasks.add_task(
                generate_conversation_title_langchain, 
                session_id, 
                message.message
            )
        
        # Prepare response
        response = ChatResponse(
            response=ai_response["response"],
            model_used=ai_response["model_used"], 
            status="success",
            session_id=session_id,
            generation_time=ai_response["response_time"],
            tokens_used=int(ai_response.get("tokens_estimated", 0))
        )
        
        return response
        
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Chat error: {str(e)}")
        
        return ChatResponse(
            response=f"I apologize, but I encountered an error: {str(e)}",
            model_used="error",
            status="error",
            session_id=session_id or "unknown"
        )

@router.get("/conversations")
async def list_conversations():
    """Get list of all conversation sessions with metadata."""
    
    conversation_list = []
    for session_id, conv in conversations.items():
        conversation_list.append({
            "session_id": session_id,
            "title": conv.title or f"Conversation {session_id[:8]}...",
            "message_count": len(conv.messages),
            "created_at": conv.created_at.isoformat(),
            "last_updated": conv.last_updated.isoformat(),
            "last_message_preview": (
                conv.messages[-1].content[:100] + "..." 
                if conv.messages else "No messages"
            )
        })
    
    return {
        "conversations": sorted(
            conversation_list, 
            key=lambda x: x["last_updated"], 
            reverse=True
        ),
        "total_conversations": len(conversation_list)
    }

@router.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Get full conversation history for a session."""
    
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = conversations[session_id]
    
    return {
        "session_id": session_id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "last_updated": conversation.last_updated.isoformat(),
        "messages": [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in conversation.messages
        ]
    }

@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation session."""
    
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[session_id]
    
    return {"message": f"Conversation {session_id} deleted successfully"}

@router.get("/health")
async def chat_service_health():
    """Check health of chat-related services."""
    
    ollama_status = await ollama_service.health_check()
    langchain_status = await langchain_service.health_check()
    
    return {
        "chat_service": "✅ Healthy",
        "ollama_service": ollama_status,
        "langchain_service": langchain_status,
        "active_conversations": len(conversations),
        "performance": {
            "ollama": ollama_service.get_performance_stats(),
            "langchain": langchain_service.get_performance_stats()
        },
        "timestamp": datetime.now().isoformat()
    }

# Background task functions
async def generate_conversation_title_langchain(session_id: str, first_message: str):
    """
    Generate a conversation title using advanced LangChain capabilities.
    """
    try:
        if session_id in conversations:
            title = await langchain_service.generate_conversation_title(first_message)
            conversations[session_id].title = title
            
    except Exception as e:
        # Fallback title generation
        conversations[session_id].title = first_message[:30] + "..." if len(first_message) > 30 else first_message