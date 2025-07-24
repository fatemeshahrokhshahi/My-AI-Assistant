# app/services/ollama_service.py - Enhanced Ollama service with LangChain integration

import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, AsyncGenerator
from app.config import settings
from langchain_ollama import OllamaLLM
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

class OllamaService:
    """
    Enhanced Ollama service with async support and LangChain integration.
    
    This service handles:
    - Basic chat completions
    - Streaming responses
    - Context-aware conversations
    - Health monitoring
    - Performance tracking
    """
    
    def __init__(self):
        self.base_url = settings.OLLAMA_URL
        self.default_model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        
        # Initialize LangChain LLM
        self.llm = OllamaLLM(
            base_url=self.base_url,
            model=self.default_model,
            temperature=0.1,  # Low temperature for factual responses
            top_p=0.9,
            num_predict=2000,  # Max tokens to generate
            stop=["Human:", "Assistant:", "\n\n---"]  # Stop sequences
        )
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        
    async def health_check(self) -> str:
        """Check if Ollama service is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        return f"✅ Healthy ({len(models)} models available)"
                    else:
                        return f"⚠️ Service responding but unhealthy (status: {response.status})"
        except asyncio.TimeoutError:
            return "⏱️ Service timeout - may be overloaded"
        except Exception as e:
            return f"❌ Service unavailable: {str(e)}"
    
    async def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                "name": model["name"],
                                "size": model.get("size", "unknown"),
                                "modified": model.get("modified_at", "unknown")
                            }
                            for model in data.get("models", [])
                        ]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def _prepare_system_prompt(self, use_rag: bool = False) -> str:
        """Prepare system prompt based on context"""
        if use_rag:
            return """You are an AI research assistant specializing in academic literature analysis. 
            You help users understand and explore peer-reviewed research papers from reputable journals.

            Your responsibilities:
            - Provide accurate, evidence-based answers using ONLY the provided context
            - Always cite your sources with specific paper titles and DOIs when available  
            - If information isn't in the provided context, clearly state this limitation
            - Maintain academic rigor and avoid speculation
            - Format responses clearly with proper citations
            
            Remember: You must ONLY use information from the provided research papers. 
            Do not add information from your general knowledge."""
        else:
            return """You are a helpful AI assistant with expertise in academic research and analysis.
            Provide clear, accurate, and well-structured responses.
            When discussing academic topics, emphasize the importance of peer-reviewed sources."""
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        use_rag: bool = False,
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> Dict[str, any]:
        """
        Generate a chat completion with optional RAG context.
        
        Args:
            messages: List of conversation messages
            context: Retrieved context from RAG system
            use_rag: Whether this is a RAG-enhanced query
            model: Specific model to use (optional)
            temperature: Response randomness (0.0 = deterministic)
            
        Returns:
            Response with metadata
        """
        start_time = time.time()
        
        try:
            # Prepare the full prompt
            system_prompt = self._prepare_system_prompt(use_rag)
            
            # Build the conversation
            conversation = [{"role": "system", "content": system_prompt}]
            
            # Add RAG context if provided
            if context and use_rag:
                context_message = {
                    "role": "system", 
                    "content": f"""Here is the relevant research context to answer the user's question:

                    RESEARCH CONTEXT:
                    {context}
                    
                    Please answer the user's question based ONLY on this context. 
                    Cite specific papers when making claims."""
                }
                conversation.append(context_message)
            
            # Add conversation history
            conversation.extend(messages)
            
            # Update LLM settings if needed
            if model and model != self.default_model:
                self.llm.model = model
            self.llm.temperature = temperature
            
            # Generate response using LangChain
            formatted_messages = self._format_messages_for_langchain(conversation)
            response_text = await self._async_generate(formatted_messages)
            
            # Calculate metrics
            response_time = time.time() - start_time
            self.request_count += 1
            self.total_response_time += response_time
            
            return {
                "response": response_text,
                "model_used": f"ollama/{model or self.default_model}",
                "response_time": response_time,
                "tokens_estimated": len(response_text.split()) * 1.3,  # Rough estimate
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
    
    def _format_messages_for_langchain(self, messages: List[Dict[str, str]]) -> str:
        """Convert message format to a single prompt for LangChain"""
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        formatted_parts.append("Assistant:")  # Prompt for response
        return "\n\n".join(formatted_parts)
    
    async def _async_generate(self, prompt: str) -> str:
        """Generate response asynchronously"""
        loop = asyncio.get_event_loop()
        # Run the synchronous LangChain call in a thread pool
        response = await loop.run_in_executor(None, self.llm.invoke, prompt)
        return response
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        use_rag: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion token by token.
        
        This is useful for real-time chat interfaces.
        """
        try:
            # Prepare the prompt (similar to chat_completion)
            system_prompt = self._prepare_system_prompt(use_rag)
            conversation = [{"role": "system", "content": system_prompt}]
            
            if context and use_rag:
                context_message = {
                    "role": "system",
                    "content": f"RESEARCH CONTEXT:\n{context}\n\nAnswer based only on this context."
                }
                conversation.append(context_message)
                
            conversation.extend(messages)
            prompt = self._format_messages_for_langchain(conversation)
            
            # Use callback handler for streaming
            callback_handler = StreamingCallbackHandler()
            
            # Generate with streaming (this is a simplified version - 
            # real streaming would require more complex implementation)
            response = await self._async_generate(prompt)
            
            # Simulate streaming by yielding words
            words = response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)  # Small delay for streaming effect
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get service performance statistics"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "average_response_time": round(avg_response_time, 3),
            "total_response_time": round(self.total_response_time, 3)
        }
    
    async def simple_query(self, query: str, model: Optional[str] = None) -> str:
        """
        Simple query method for quick testing (like your original function).
        """
        result = await self.chat_completion(
            messages=[{"role": "user", "content": query}],
            model=model
        )
        return result["response"]