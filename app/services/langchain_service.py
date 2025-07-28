# app/services/langchain_service.py - Simple, Working LangChain Service

import asyncio
import time
from typing import Dict, List, Optional, Any

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from app.config import settings
from app.models.chat_models import MessageRole, ConversationMessage

class LangChainService:
    """
    Simple, working LangChain service without complex memory objects.
    This avoids the compatibility issues while still providing enhanced AI capabilities.
    """
    
    def __init__(self):
        # Initialize LangChain LLM
        self.llm = OllamaLLM(
            base_url=settings.OLLAMA_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.1,
            num_predict=2000,
            timeout=30
        )
        
        # Simple prompt template
        self.conversation_template = PromptTemplate(
            input_variables=["history", "question"],
            template="""You are an intelligent AI research assistant with expertise in academic literature.

Your responses should be accurate, clear, professional, and helpful.

Conversation history:
{history}

Current question: {question}

Response:"""
        )
        
        # Title generation template
        self.title_template = PromptTemplate(
            input_variables=["message"],
            template="""Generate a short, descriptive title (maximum 6 words) for a conversation starting with: "{message}"

Examples:
- "How does machine learning work?" → "Machine Learning Fundamentals"
- "What is climate change?" → "Climate Change Overview" 
- "Help with statistics" → "Statistical Analysis Help"

Title:"""
        )
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
    
    def _format_conversation_history(self, conversation_history: List[ConversationMessage]) -> str:
        """Convert conversation history to a simple string format"""
        if not conversation_history:
            return "No previous conversation."
        
        history_text = ""
        for msg in conversation_history[-10:]:  # Keep last 10 messages
            if msg.role == MessageRole.USER:
                history_text += f"Human: {msg.content}\n"
            elif msg.role == MessageRole.ASSISTANT:
                history_text += f"Assistant: {msg.content}\n"
        
        return history_text.strip() if history_text else "No previous conversation."
    
    async def chat_with_memory(
        self,
        message: str,
        session_id: str,
        conversation_history: List[ConversationMessage] = None,
        use_summary_memory: bool = False
    ) -> Dict[str, Any]:
        """
        Chat with conversation memory using simple string formatting.
        """
        start_time = time.time()
        
        try:
            # Format conversation history
            history_text = self._format_conversation_history(conversation_history or [])
            
            # Create the full prompt
            prompt = self.conversation_template.format(
                history=history_text,
                question=message
            )
            
            # Generate response
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.llm.invoke,
                prompt
            )
            
            # Calculate metrics
            response_time = time.time() - start_time
            self.request_count += 1
            self.total_response_time += response_time
            
            return {
                "response": response,
                "model_used": f"langchain/{self.llm.model}",
                "response_time": response_time,
                "tokens_estimated": len(response.split()) * 1.3,  # Rough estimate
                "memory_type": "simple_history",
                "session_id": session_id,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "model_used": "error",
                "response_time": time.time() - start_time,
                "status": "error",
                "error": str(e)
            }
    
    async def chat_with_context(
        self,
        message: str,
        context: str = None,
        conversation_history: List[ConversationMessage] = None
    ) -> Dict[str, Any]:
        """
        Chat with additional context (for future RAG integration).
        """
        start_time = time.time()
        
        try:
            # Format conversation history
            history_text = self._format_conversation_history(conversation_history or [])
            
            # Create prompt with context if provided
            if context:
                prompt = f"""You are an expert research assistant. Use the provided context to answer the question accurately.

Context: {context}

Conversation history:
{history_text}

Question: {message}

Response:"""
            else:
                prompt = self.conversation_template.format(
                    history=history_text,
                    question=message
                )
            
            # Generate response
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.llm.invoke,
                prompt
            )
            
            response_time = time.time() - start_time
            
            return {
                "response": response,
                "model_used": f"langchain/{self.llm.model}",
                "response_time": response_time,
                "tokens_estimated": len(response.split()) * 1.3,
                "context_used": bool(context),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "model_used": "error",
                "response_time": time.time() - start_time,
                "status": "error",
                "error": str(e)
            }
    
    async def generate_conversation_title(self, first_message: str) -> str:
        """Generate an intelligent conversation title"""
        try:
            prompt = self.title_template.format(message=first_message[:200])
            
            loop = asyncio.get_event_loop()
            title = await loop.run_in_executor(
                None,
                self.llm.invoke,
                prompt
            )
            
            # Clean up the generated title
            title = title.strip().replace('"', '').replace("Title:", "").strip()
            if len(title) > 50:
                title = first_message[:47] + "..."
            
            return title
            
        except Exception as e:
            # Fallback title generation
            return first_message[:30] + "..." if len(first_message) > 30 else first_message
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "average_response_time": round(avg_response_time, 3),
            "service_type": "simple_langchain",
            "features": ["conversation_history", "context_support", "title_generation"]
        }
    
    async def health_check(self) -> str:
        """Check health of LangChain service"""
        try:
            # Test basic LLM functionality
            test_response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    self.llm.invoke,
                    "Hello"
                ),
                timeout=10
            )
            
            if test_response and len(test_response.strip()) > 0:
                return "✅ LangChain service healthy"
            else:
                return "⚠️ LangChain service responding but slow"
            
        except asyncio.TimeoutError:
            return "⏱️ LangChain service timeout"
        except Exception as e:
            return f"❌ LangChain service error: {str(e)}"