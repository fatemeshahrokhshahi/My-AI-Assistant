# app/routers/rag.py - RAG system router for document management and intelligent queries

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import shutil
from pathlib import Path

from app.models.chat_models import RAGQuery, RAGResponse, DocumentMetadata
from app.services.rag_service import RAGService
from app.config import settings

# Create router
router = APIRouter()

# Initialize RAG service
rag_service = RAGService()

@router.post("/query", response_model=RAGResponse)
async def rag_query(query: RAGQuery):
    """
    Intelligent query using RAG system - the main endpoint for your research assistant!
    
    This endpoint:
    1. Takes a research question
    2. Searches your knowledge base of academic papers
    3. Generates an intelligent answer with proper citations
    
    Perfect for questions like:
    - "What are the latest developments in machine learning?"
    - "How does climate change affect biodiversity?"
    - "What are the applications of CRISPR technology?"
    """
    try:
        # Perform RAG query
        response = await rag_service.query_knowledge_base(
            query=query.query,
            k=query.k,
            similarity_threshold=query.similarity_threshold,
            filter_by_source=query.filter_metadata.get("source") if query.filter_metadata else None
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing RAG query: {str(e)}"
        )

@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a document (PDF, JSON, TXT) to the knowledge base.
    
    The document will be processed and indexed for future RAG queries.
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.json', '.txt'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Check file size
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file.size} bytes (max: {settings.MAX_FILE_SIZE})"
            )
        
        # Save uploaded file
        upload_path = Path(settings.UPLOAD_DIR) / file.filename
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document and add to knowledge base
        success, message, metadata = await rag_service.add_document_to_knowledge_base(
            file_path=str(upload_path),
            filename=file.filename,
            document_type=file_extension[1:]  # Remove the dot
        )
        
        # Clean up uploaded file
        if upload_path.exists():
            upload_path.unlink()
        
        if success:
            return {
                "success": True,
                "message": message,
                "document_metadata": metadata.dict() if metadata else None
            }
        else:
            raise HTTPException(status_code=500, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )



@router.post("/add-springer-by-upload")
async def add_springer_by_upload(file: UploadFile = File(...)):
    """
    Alternative: Upload your Springer JSON and process it correctly
    This will treat it as a dataset, not a single document
    """
    try:
        # Save the uploaded file temporarily
        temp_path = f"./temp_{file.filename}"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process as Springer dataset (multiple papers)
        successful, failed, errors = await rag_service.add_springer_dataset_to_knowledge_base(
            temp_path
        )
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": True,
            "message": f"Processed Springer dataset via upload",
            "results": {
                "successful_papers": successful,
                "failed_papers": failed,
                "total_papers": successful + failed,
                "errors": errors[:5] if errors else []  # Show first 5 errors
            }
        }
        
    except Exception as e:
        # Clean up temp file on error
        temp_path = f"./temp_{file.filename}"
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "success": False,
            "error": str(e),
            "suggestion": "Check that your JSON file contains an array of papers"
        }
    



@router.get("/stats")
async def get_knowledge_base_stats():
    """
    Get comprehensive statistics about the knowledge base.
    
    Shows:
    - Number of documents indexed
    - Available sources (Springer, IEEE, etc.)
    - Processing performance metrics
    - RAG query statistics
    """
    try:
        stats = await rag_service.get_knowledge_base_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting knowledge base stats: {str(e)}"
        )

@router.get("/health")
async def rag_system_health():
    """Check health of the entire RAG system"""
    try:
        health_status = await rag_service.health_check()
        
        return {
            "rag_system": health_status,
            "timestamp": str(Path(__file__).stat().st_mtime),
            "components": {
                "document_processor": "✅ Ready",
                "vector_database": "✅ ChromaDB operational", 
                "langchain_service": "✅ Advanced AI ready",
                "embedding_model": f"✅ {settings.EMBEDDING_MODEL}"
            }
        }
        
    except Exception as e:
        return {
            "rag_system": f"❌ Error: {str(e)}",
            "status": "unhealthy"
        }



@router.delete("/reset-knowledge-base")
async def reset_knowledge_base():
    """
    Reset the entire knowledge base (use with caution!).
    
    This will delete all indexed documents and embeddings.
    Only use this if you want to start fresh.
    """
    try:
        # Reset vector database
        success = rag_service.vector_db.reset_collection()
        
        if success:
            return {
                "success": True,
                "message": "Knowledge base reset successfully",
                "warning": "All documents and embeddings have been deleted"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reset knowledge base"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting knowledge base: {str(e)}"
        )

# Background task functions
async def process_springer_dataset_background(json_file_path: str):
    """
    Background task to process Springer dataset.
    
    This runs asynchronously so it doesn't block the API.
    """
    try:
        print(f"🚀 Starting background processing of Springer dataset: {json_file_path}")
        
        successful, failed, errors = await rag_service.add_springer_dataset_to_knowledge_base(
            json_file_path
        )
        
        print(f"✅ Springer dataset processing completed:")
        print(f"   Successful papers: {successful}")
        print(f"   Failed papers: {failed}")
        
        if errors:
            print("❌ Errors encountered:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
        
    except Exception as e:
        print(f"❌ Background processing error: {str(e)}")

@router.get("/search-papers")
async def search_papers(
    query: str,
    limit: int = 10,
    source: Optional[str] = None,
    min_similarity: float = 0.7
):
    """
    Search for specific papers in the knowledge base.
    
    Returns paper metadata and relevance scores without generating full answers.
    Useful for exploring what's available in your knowledge base.
    """
    try:
        rag_result = await rag_service.vector_db.search_with_text_query(
            query_text=query,
            embedding_function=rag_service.document_processor.get_embedding,
            k=limit,
            similarity_threshold=min_similarity,
            filter_by_source=source
        )
        
        # Format results for easy browsing
        papers = []
        for i, chunk in enumerate(rag_result.retrieved_chunks):
            paper_info = {
                "rank": i + 1,
                "title": chunk.metadata.get("document_title", "Unknown Title"),
                "source": chunk.metadata.get("document_source", "Unknown Source"),
                "doi": chunk.metadata.get("doi", ""),
                "similarity_score": rag_result.similarity_scores[i] if i < len(rag_result.similarity_scores) else 0,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            }
            papers.append(paper_info)
        
        return {
            "query": query,
            "total_results": len(papers),
            "retrieval_time": rag_result.retrieval_time,
            "papers": papers
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching papers: {str(e)}"
        )
   
    
    


