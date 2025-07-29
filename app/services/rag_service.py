# Complete RAG (Retrieval-Augmented Generation) system

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.config import settings
from app.models.chat_models import (
    RAGQuery, RAGResult, RAGResponse, DocumentMetadata, 
    DocumentChunk, ConversationMessage
)
from app.services.document_service import DocumentProcessor
from app.services.vector_service import VectorDatabaseService
from app.services.langchain_service import LangChainService

class RAGService:
    """
    Complete RAG (Retrieval-Augmented Generation) system.
    
    This is the core of your AI research assistant! It:
    1. Takes user questions
    2. Retrieves relevant document chunks from your knowledge base
    3. Generates intelligent answers using the retrieved context
    4. Provides proper citations and source attribution
    
    Perfect for querying your Springer academic papers!
    """
    
    def __init__(self):
        # Initialize all component services
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabaseService()
        self.langchain_service = LangChainService()
        
        # RAG configuration
        self.default_k = settings.RETRIEVAL_K
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        # Performance tracking
        self.rag_query_count = 0
        self.total_rag_time = 0.0
        
        print("✅ RAG Service initialized successfully")
    
    async def query_knowledge_base(
        self,
        query: str,
        k: int = None,
        similarity_threshold: float = None,
        filter_by_source: str = None,
        conversation_history: List[ConversationMessage] = None
    ) -> RAGResponse:
        """
        Main RAG query method - this is what makes your assistant intelligent!
        
        Args:
            query: User's research question
            k: Number of relevant documents to retrieve
            similarity_threshold: Minimum relevance score
            filter_by_source: Optional filter (e.g., "Springer")
            conversation_history: Previous conversation for context
            
        Returns:
            Complete RAG response with answer and citations
        """
        start_time = time.time()
        
        try:
            # Use defaults if not specified
            k = k or self.default_k
            similarity_threshold = similarity_threshold or self.similarity_threshold
            
            print(f"🔍 RAG Query: '{query}' (k={k}, threshold={similarity_threshold})")
            
            # Step 1: Retrieve relevant document chunks
            retrieval_start = time.time()
            rag_result = await self.vector_db.search_with_text_query(
                query_text=query,
                embedding_function=self.document_processor.get_embedding,
                k=k,
                similarity_threshold=similarity_threshold,
                filter_by_source=filter_by_source
            )
            retrieval_time = time.time() - retrieval_start
            
            print(f"📊 Retrieved {len(rag_result.retrieved_chunks)} relevant chunks in {retrieval_time:.3f}s")
            
            # Step 2: Generate response using retrieved context
            generation_start = time.time()
            
            if rag_result.retrieved_chunks:
                # Create context from retrieved chunks
                context = self._create_context_from_chunks(rag_result.retrieved_chunks)
                
                # Generate AI response with context
                ai_response = await self.langchain_service.chat_with_context(
                    message=query,
                    context=context,
                    conversation_history=conversation_history
                )
                
                answer = ai_response["response"]
                
                # Extract source documents
                sources = self._extract_source_documents(rag_result.retrieved_chunks)
                
            else:
                # No relevant documents found
                answer = """I couldn't find relevant information in the knowledge base to answer your question. 
                
                This might be because:
                - The question is outside the scope of the available academic papers
                - The similarity threshold is too high
                - The knowledge base needs more relevant documents
                
                Try rephrasing your question or lowering the similarity threshold."""
                
                sources = []
                rag_result.retrieved_chunks = []
            
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            
            # Update performance stats
            self.rag_query_count += 1
            self.total_rag_time += total_time
            
            # Create comprehensive RAG response
            response = RAGResponse(
                query=query,
                answer=answer,
                sources=sources,
                chunks_used=rag_result.retrieved_chunks,
                similarity_scores=rag_result.similarity_scores,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                confidence_score=self._calculate_confidence_score(rag_result.similarity_scores)
            )
            
            print(f"✅ RAG query completed in {total_time:.3f}s")
            return response
            
        except Exception as e:
            print(f"❌ RAG query error: {str(e)}")
            
            # Return error response
            return RAGResponse(
                query=query,
                answer=f"I apologize, but I encountered an error while searching the knowledge base: {str(e)}",
                sources=[],
                chunks_used=[],
                similarity_scores=[],
                retrieval_time=0.0,
                generation_time=0.0,
                confidence_score=0.0
            )
    
    async def add_document_to_knowledge_base(
        self,
        file_path: str,
        filename: str,
        document_type: str = "pdf"
    ) -> Tuple[bool, str, Optional[DocumentMetadata]]:
        """
        Add a new document to the knowledge base.
        
        Args:
            file_path: Path to the document file
            filename: Original filename
            document_type: Type of document (pdf, json, txt)
            
        Returns:
            Tuple of (success, message, document_metadata)
        """
        try:
            print(f"📄 Processing document: {filename}")
            
            # Step 1: Process the document
            from app.models.chat_models import DocumentType
            doc_type = DocumentType(document_type.lower())
            
            metadata, chunks = await self.document_processor.process_uploaded_file(
                file_path=file_path,
                filename=filename,
                file_type=doc_type
            )
            
            if not chunks:
                return False, f"Failed to process document: {metadata.error_message}", metadata
            
            print(f"📊 Created {len(chunks)} chunks from document")
            
            # Step 2: Generate embeddings for all chunks
            print("🔄 Generating embeddings...")
            embeddings = []
            
            for chunk in chunks:
                embedding = await self.document_processor.get_embedding(chunk.content)
                embeddings.append(embedding)
            
            print(f"✅ Generated {len(embeddings)} embeddings")
            
            # Step 3: Store in vector database
            success = await self.vector_db.add_document_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_metadata=metadata
            )
            
            if success:
                message = f"Successfully added '{filename}' to knowledge base ({len(chunks)} chunks)"
                print(f"✅ {message}")
                return True, message, metadata
            else:
                return False, "Failed to store document in vector database", metadata
                
        except Exception as e:
            error_msg = f"Error adding document to knowledge base: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg, None
    
    async def add_springer_dataset_to_knowledge_base(
        self, 
        json_file_path: str
    ) -> Tuple[int, int, List[str]]:
        """
        Add your existing Springer JSON dataset to the knowledge base!
        
        This method will process all papers from your previous research.
        
        Args:
            json_file_path: Path to your Springer JSON file
            
        Returns:
            Tuple of (successful_papers, failed_papers, error_messages)
        """
        try:
            print(f"📚 Processing Springer dataset: {json_file_path}")
            
           
            paper_results = await self.document_processor.process_springer_json(json_file_path)
            
            successful_papers = 0
            failed_papers = 0
            error_messages = []
            
            for metadata, chunks in paper_results:
                try:
                    if not chunks:
                        failed_papers += 1
                        error_messages.append(f"No chunks created for: {metadata.title}")
                        continue
                    
                    # Generate embeddings for chunks
                    embeddings = []
                    for chunk in chunks:
                        embedding = await self.document_processor.get_embedding(chunk.content)
                        embeddings.append(embedding)
                    
                    # Add to vector database
                    success = await self.vector_db.add_document_chunks(
                        chunks=chunks,
                        embeddings=embeddings,
                        document_metadata=metadata
                    )
                    
                    if success:
                        successful_papers += 1
                        print(f"✅ Added paper: {metadata.title[:50]}...")
                    else:
                        failed_papers += 1
                        error_messages.append(f"Failed to store: {metadata.title}")
                        
                except Exception as e:
                    failed_papers += 1
                    error_messages.append(f"Error processing {metadata.title}: {str(e)}")
            
            print(f"📊 Springer dataset processing complete:")
            print(f"   ✅ Successful: {successful_papers} papers")
            print(f"   ❌ Failed: {failed_papers} papers")
            
            return successful_papers, failed_papers, error_messages
            
        except Exception as e:
            error_msg = f"Error processing Springer dataset: {str(e)}"
            print(f"❌ {error_msg}")
            return 0, 1, [error_msg]
    
    def _create_context_from_chunks(self, chunks: List[DocumentChunk]) -> str:
        """
        Create a well-formatted context string from retrieved chunks.
        
        This is crucial for RAG - how we present the retrieved information to the AI.
        """
        if not chunks:
            return "No relevant information found."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Get document info from metadata
            doc_title = chunk.metadata.get("document_title", "Unknown Document")
            doc_source = chunk.metadata.get("document_source", "Unknown Source")
            doi = chunk.metadata.get("doi", "")
            
            # Format each chunk with proper attribution
            chunk_header = f"**Source {i+1}: {doc_title}**"
            if doi:
                chunk_header += f" (DOI: {doi})"
            chunk_header += f" - {doc_source}"
            
            context_part = f"{chunk_header}\n{chunk.content}\n"
            context_parts.append(context_part)
        
        context = "\n---\n".join(context_parts)
        
        # Add instruction for the AI
        instruction = """Based on the above academic sources, please provide a comprehensive answer to the user's question. 

Important instructions:
- Use only the information provided in the sources above
- When making specific claims, reference the source (e.g., "According to Source 1...")
- If the sources don't contain sufficient information, clearly state this limitation
- Maintain academic rigor and avoid speculation
- Include relevant details like study findings, methodologies, or key concepts mentioned in the sources

Sources provided above:
"""
        
        return instruction + "\n\n" + context
    
    def _extract_source_documents(self, chunks: List[DocumentChunk]) -> List[DocumentMetadata]:
        """Extract unique source documents from retrieved chunks"""
        seen_docs = set()
        sources = []
        
        for chunk in chunks:
            doc_id = chunk.document_id
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                
                # Create DocumentMetadata from chunk metadata
                metadata = DocumentMetadata(
                    filename=chunk.metadata.get("document_title", "Unknown") + ".pdf",
                    file_size=0,  # Not available from chunk
                    document_type=chunk.metadata.get("document_type", "pdf"),
                    title=chunk.metadata.get("document_title"),
                    authors=[],  # Could be parsed from chunk metadata if available
                    abstract=chunk.metadata.get("abstract", ""),
                    doi=chunk.metadata.get("doi"),
                    source=chunk.metadata.get("document_source")
                )
                sources.append(metadata)
        
        return sources
    
    def _calculate_confidence_score(self, similarity_scores: List[float]) -> float:
        """
        Calculate confidence score based on similarity scores.
        
        Higher similarity scores and more consistent scores = higher confidence
        """
        if not similarity_scores:
            return 0.0
        
        # Use average similarity score as base confidence
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Boost confidence if we have multiple high-quality sources
        if len(similarity_scores) >= 3 and avg_similarity > 0.8:
            return min(avg_similarity * 1.1, 1.0)
        
        return avg_similarity
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base"""
        try:
            # Get vector database stats
            vector_stats = self.vector_db.get_collection_stats()
            
            # Get document processing stats
            doc_stats = self.document_processor.get_processing_stats()
            
            # Calculate RAG performance stats
            avg_rag_time = (
                self.total_rag_time / self.rag_query_count 
                if self.rag_query_count > 0 else 0
            )
            
            return {
                "knowledge_base": {
                    "total_documents": vector_stats.get("total_chunks", 0),
                    "available_sources": vector_stats.get("available_sources", []),
                    "document_types": vector_stats.get("document_types", [])
                },
                "processing": doc_stats,
                "rag_performance": {
                    "total_queries": self.rag_query_count,
                    "average_query_time": round(avg_rag_time, 3),
                    "default_retrieval_k": self.default_k,
                    "similarity_threshold": self.similarity_threshold
                },
                "vector_database": vector_stats
            }
            
        except Exception as e:
            return {"error": f"Failed to get knowledge base stats: {str(e)}"}
    
    async def health_check(self) -> str:
        """Check health of the entire RAG system"""
        try:
            # Check all component services
            doc_status = "✅ Document processor ready"
            vector_status = await self.vector_db.health_check()
            langchain_status = await self.langchain_service.health_check()
            
            # Check if knowledge base has content
            stats = self.vector_db.get_collection_stats()
            doc_count = stats.get("total_chunks", 0)
            
            if doc_count > 0:
                return f"✅ RAG system fully operational ({doc_count} documents indexed)"
            else:
                return "⚠️ RAG system ready but knowledge base is empty"
                
        except Exception as e:
            return f"❌ RAG system error: {str(e)}"