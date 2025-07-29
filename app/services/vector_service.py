import asyncio
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import time

from app.config import settings
from app.models.chat_models import DocumentChunk, DocumentMetadata, RAGQuery, RAGResult

class VectorDatabaseService:
    """
    ChromaDB vector database service for semantic search.
    
    This service handles:
    - Storing document embeddings
    - Semantic similarity search
    - Metadata filtering
    - Collection management
    """
    
    def __init__(self):
        self.db_path = settings.VECTOR_DB_PATH
        self.collection_name = "academic_papers"
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection for academic papers
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Academic papers and research documents"}
            )
            
            print(f"✅ ChromaDB initialized at {self.db_path}")
            print(f"📊 Collection '{self.collection_name}' ready with {self.collection.count()} documents")
            
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {str(e)}")
            raise
    
    async def add_document_chunks(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: List[List[float]],
        document_metadata: DocumentMetadata
    ) -> bool:
        """
        Add document chunks with their embeddings to the vector database.
        """
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = []
            
            # Create metadata for each chunk
            for chunk in chunks:
                chunk_metadata = {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "document_title": document_metadata.title or "Unknown Title",
                    "document_source": document_metadata.source or "Unknown Source",
                    "document_type": document_metadata.document_type.value,
                    "upload_time": document_metadata.upload_time.isoformat(),
                    "authors": json.dumps(document_metadata.authors),
                    "keywords": json.dumps(document_metadata.keywords),
                    "doi": document_metadata.doi or "",
                    "abstract": document_metadata.abstract or ""
                }
                metadatas.append(chunk_metadata)
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"✅ Added {len(chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            print(f"❌ Error adding chunks to vector database: {str(e)}")
            return False
    
    async def semantic_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using query embedding.
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key in ["document_source", "document_type", "document_title"]:
                        where_clause[key] = value
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1.0 - distance
                    
                    search_result = {
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    }
                    search_results.append(search_result)
            
            # Update performance stats
            query_time = asyncio.get_event_loop().time() - start_time
            self.query_count += 1
            self.total_query_time += query_time
            
            return search_results
            
        except Exception as e:
            print(f"❌ Error in semantic search: {str(e)}")
            return []
    
    async def search_with_text_query(
        self,
        query_text: str,
        embedding_function,
        k: int = 5,
        similarity_threshold: float = 0.7,
        filter_by_source: str = None
    ) -> RAGResult:
        """
        Search using text query (will be embedded automatically).
        
        PROPERLY INDENTED INSIDE THE CLASS!
        """
        start_time = time.time()
        
        try:
            print(f"🔍 RAG Search: '{query_text}' (k={k}, threshold={similarity_threshold})")
            
            # Generate embedding for query
            query_embedding = await embedding_function(query_text)
            print(f"📊 Generated query embedding: {len(query_embedding)} dimensions")
            
            # Prepare where clause for filtering
            where_clause = None
            if filter_by_source:
                where_clause = {"document_source": filter_by_source}
            
            # Query ChromaDB directly
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 2,  # Get more results to account for threshold filtering
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"📋 ChromaDB returned {len(results['documents'][0]) if results.get('documents') else 0} raw results")
            
            # Process results and filter by threshold
            retrieved_chunks = []
            similarity_scores = []
            
            if results.get("documents") and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert ChromaDB distance to similarity score
                    # ChromaDB uses cosine distance, so similarity = 1 - (distance / 2)
                    similarity = 1.0 - (distance / 2.0)
                    
                    print(f"   Result {i}: similarity = {similarity:.3f}, distance = {distance:.3f}")
                    
                    if similarity >= similarity_threshold:
                        # Create DocumentChunk object
                        chunk = DocumentChunk(
                            chunk_id=f"search_{i}_{metadata.get('document_id', 'unknown')}",
                            document_id=metadata.get("document_id", "unknown"),
                            content=doc,
                            chunk_index=metadata.get("chunk_index", i),
                            start_char=metadata.get("start_char", 0),
                            end_char=metadata.get("end_char", len(doc)),
                            metadata=metadata
                        )
                        
                        retrieved_chunks.append(chunk)
                        similarity_scores.append(similarity)
            
            retrieval_time = time.time() - start_time
            
            print(f"✅ Found {len(retrieved_chunks)} chunks above threshold {similarity_threshold}")
            
            return RAGResult(
                query=query_text,
                retrieved_chunks=retrieved_chunks,
                similarity_scores=similarity_scores,
                retrieval_time=retrieval_time,
                total_results=len(results["documents"][0]) if results.get("documents") else 0
            )
            
        except Exception as e:
            print(f"❌ Error in search_with_text_query: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return RAGResult(
                query=query_text,
                retrieved_chunks=[],
                similarity_scores=[],
                retrieval_time=time.time() - start_time,
                total_results=0
            )

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            
            if results['documents']:
                return {
                    "document_id": document_id,
                    "chunks": results['documents'],
                    "metadatas": results['metadatas']
                }
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving document {document_id}: {str(e)}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection"""
        try:
            count = self.collection.count()
            
            # Get a sample to analyze metadata
            sample = self.collection.peek(limit=min(100, count))
            
            # Analyze sources and types
            sources = set()
            doc_types = set()
            
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    if 'document_source' in metadata:
                        sources.add(metadata['document_source'])
                    if 'document_type' in metadata:
                        doc_types.add(metadata['document_type'])
            
            avg_query_time = (
                self.total_query_time / self.query_count 
                if self.query_count > 0 else 0
            )
            
            return {
                "total_chunks": count,
                "available_sources": list(sources),
                "document_types": list(doc_types),
                "collection_name": self.collection_name,
                "database_path": self.db_path,
                "query_performance": {
                    "total_queries": self.query_count,
                    "average_query_time": round(avg_query_time, 4)
                }
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get stats: {str(e)}",
                "total_chunks": 0
            }
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"✅ Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                print(f"⚠️ No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting document {document_id}: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (use with caution!)"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Academic papers and research documents"}
            )
            print("✅ Collection reset successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error resetting collection: {str(e)}")
            return False
    
    async def health_check(self) -> str:
        """Check health of vector database service"""
        try:
            # Test basic operations
            count = self.collection.count()
            
            # Test a simple query if we have data
            if count > 0:
                test_results = self.collection.peek(limit=1)
                if test_results['documents']:
                    return f"✅ Vector DB healthy ({count} documents)"
            
            return f"✅ Vector DB ready (empty collection)"
            
        except Exception as e:
            return f"❌ Vector DB error: {str(e)}"