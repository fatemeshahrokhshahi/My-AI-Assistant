# app/models/chat_models.py - Enhanced models based on your original ones

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# === Enhanced versions of your original models ===

class ChatMessage(BaseModel):
    """
    Enhanced version of your original ChatMessage.
    
    Your original had: message: str
    Enhanced version adds session management and RAG options.
    """
    message: str = Field(..., min_length=1, max_length=10000, description="The user's message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    use_rag: bool = Field(True, description="Whether to use RAG for journal paper queries")
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000, description="Maximum tokens in response")
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        return v.strip()

class ChatResponse(BaseModel):
    """
    Enhanced version of your original ChatResponse.
    
    Your original had: response, model_used, status
    Enhanced version adds RAG metadata and performance tracking.
    """
    response: str = Field(..., description="The AI's response")
    model_used: str = Field(..., description="Which model generated this response")
    status: str = Field("success", description="Response status")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    
    # New RAG-specific fields for journal paper queries
    sources_used: List[str] = Field(default=[], description="Journal papers used as sources")
    similarity_scores: List[float] = Field(default=[], description="Relevance scores for sources")
    retrieval_time: Optional[float] = Field(None, description="Time spent retrieving context")
    generation_time: Optional[float] = Field(None, description="Time spent generating response")
    
    # Performance tracking
    tokens_used: Optional[int] = Field(None, description="Tokens used in generation")
    context_length: Optional[int] = Field(None, description="Length of retrieved context")

# === New models for conversation management ===

class MessageRole(str, Enum):
    """Define message roles in conversations"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationMessage(BaseModel):
    """Individual message in a conversation with metadata"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Conversation(BaseModel):
    """Complete conversation with history and metadata"""
    session_id: str
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    title: Optional[str] = Field(None, description="Auto-generated conversation title")
    
    def add_message(self, role: MessageRole, content: str, metadata: Dict[str, Any] = None):
        """Add a new message to the conversation"""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_updated = datetime.now()
    
    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation context for the AI"""
        recent_messages = self.messages[-max_messages:]
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in recent_messages
        ]

# === Models for document/journal paper processing ===

class DocumentType(str, Enum):
    """Supported document types for journal papers"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"  # For your existing Springer JSON datasets

class DocumentMetadata(BaseModel):
    """
    Metadata for uploaded journal papers and documents.
    This will work great with your existing Springer data!
    """
    filename: str
    file_size: int
    document_type: DocumentType
    upload_time: datetime = Field(default_factory=datetime.now)
    processing_status: str = "pending"  # pending, processing, completed, failed
    chunk_count: Optional[int] = Field(None, description="Number of chunks created")
    
    # Journal-specific metadata (matches your Springer data structure!)
    title: Optional[str] = Field(None, description="Paper title")
    authors: List[str] = Field(default_factory=list, description="Paper authors")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    keywords: List[str] = Field(default_factory=list, description="Paper keywords/subjects")
    doi: Optional[str] = Field(None, description="DOI identifier")
    source: Optional[str] = Field(None, description="Publisher (e.g., 'Springer', 'IEEE')")
    
    # Processing metadata
    processing_time: Optional[float] = Field(None, description="Time taken to process")
    error_message: Optional[str] = Field(None, description="Error if processing failed")

class DocumentChunk(BaseModel):
    """A chunk of text from a journal paper with metadata"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

# === RAG System Models ===

class RAGQuery(BaseModel):
    """Query for the RAG system to search journal papers"""
    query: str = Field(..., min_length=1, description="The research question")
    k: int = Field(5, ge=1, le=20, description="Number of papers to retrieve")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    filter_by_source: Optional[str] = Field(None, description="Filter by publisher (e.g., 'Springer')")

class RAGResult(BaseModel):
    """Result from RAG retrieval of journal papers"""
    query: str
    retrieved_chunks: List[DocumentChunk]
    similarity_scores: List[float]
    retrieval_time: float
    total_results: int

class RAGResponse(BaseModel):
    """Complete RAG response with generated answer and journal citations"""
    query: str
    answer: str
    sources: List[DocumentMetadata]
    chunks_used: List[DocumentChunk]
    similarity_scores: List[float]
    retrieval_time: float
    generation_time: float
    confidence_score: Optional[float] = Field(None, description="AI confidence in the answer")

# === System Status Models (enhanced versions of your original endpoints) ===

class SystemHealth(BaseModel):
    """Enhanced version of your original health check"""
    status: str
    components: Dict[str, str]
    timestamp: str
    uptime: Optional[float] = Field(None, description="System uptime in seconds")
    
class ModelInfo(BaseModel):
    """Enhanced version of your original model info"""
    name: str
    size: str
    description: str
    is_available: bool = True
    is_default: bool = False