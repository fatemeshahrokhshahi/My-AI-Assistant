# Document processing pipeline for academic papers

import json
import os
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

# Document processing imports
import pypdf
from sentence_transformers import SentenceTransformer


from app.config import settings
from app.models.chat_models import DocumentMetadata, DocumentChunk, DocumentType

class DocumentProcessor:
    """
    Advanced document processing pipeline for academic papers.
    
    Handles:
    - PDF extraction
    - JSON files (perfect for your Springer data!)
    - Text chunking with overlap
    - Metadata extraction
    - Content cleaning and normalization
    """
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.max_file_size = settings.MAX_FILE_SIZE
        
        # Initialize sentence transformer for embeddings
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("✅ Embedding model loaded successfully")
        
        # Ensure upload directories exist
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
    
    async def process_uploaded_file(
        self, 
        file_path: str, 
        filename: str,
        file_type: DocumentType
    ) -> Tuple[DocumentMetadata, List[DocumentChunk]]:
        """
        Process an uploaded file and return metadata + chunks.
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            file_type: Type of document (PDF, JSON, etc.)
            
        Returns:
            Tuple of (document_metadata, list_of_chunks)
        """
        start_time = datetime.now()
        
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Check file size limit
            if file_size > self.max_file_size:
                raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
            
            # Extract content based on file type
            if file_type == DocumentType.PDF:
                content, title, authors, abstract = await self._process_pdf(file_path)
            elif file_type == DocumentType.JSON:
                content, title, authors, abstract = await self._process_json(file_path)
            elif file_type == DocumentType.TXT:
                content, title, authors, abstract = await self._process_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Generate document ID
            document_id = self._generate_document_id(filename, content)
            
            # Create document metadata
            metadata = DocumentMetadata(
                filename=filename,
                file_size=file_size,
                document_type=file_type,
                upload_time=start_time,
                processing_status="processing",
                title=title,
                authors=authors,
                abstract=abstract
            )
            
            # Create text chunks
            chunks = await self._create_chunks(document_id, content, metadata)
            
            # Update processing status
            processing_time = (datetime.now() - start_time).total_seconds()
            metadata.processing_status = "completed"
            metadata.processing_time = processing_time
            metadata.chunk_count = len(chunks)
            
            return metadata, chunks
            
        except Exception as e:
            # Create error metadata
            error_metadata = DocumentMetadata(
                filename=filename,
                file_size=file_size if 'file_size' in locals() else 0,
                document_type=file_type,
                upload_time=start_time,
                processing_status="failed",
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            return error_metadata, []
    
    async def process_springer_json(self, json_file_path: str) -> List[Tuple[DocumentMetadata, List[DocumentChunk]]]:
        """
        Process your existing Springer JSON dataset!
        
        This method is specifically designed for your Springer data format.
        """
        results = []
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both formats: direct list or nested under 'data' key
            papers = data if isinstance(data, list) else data.get('data', [])
            
            print(f"📚 Processing {len(papers)} papers from Springer dataset...")
            
            for paper in papers:
                try:
                    # Extract paper information (matching your previous data structure)
                    title = paper.get('title', 'Untitled Paper')
                    abstract = paper.get('abstract', '')
                    subjects = paper.get('subjects', [])
                    doi = paper.get('identifier', paper.get('doi', ''))
                    source = paper.get('source', 'Springer')
                    
                    # Create content by combining title and abstract
                    content = f"Title: {title}\n\nAbstract: {abstract}"
                    if subjects:
                        content += f"\n\nKeywords/Subjects: {', '.join(subjects)}"
                    
                    # Generate document ID
                    document_id = self._generate_document_id(doi or title, content)
                    
                    # Create metadata
                    metadata = DocumentMetadata(
                        filename=f"{doi or document_id}.json",
                        file_size=len(content.encode('utf-8')),
                        document_type=DocumentType.JSON,
                        upload_time=datetime.now(),
                        processing_status="completed",
                        title=title,
                        authors=[],  # Authors not in your current format
                        abstract=abstract,
                        keywords=subjects,
                        doi=doi,
                        source=source
                    )
                    
                    # Create chunks
                    chunks = await self._create_chunks(document_id, content, metadata)
                    metadata.chunk_count = len(chunks)
                    
                    results.append((metadata, chunks))
                    
                except Exception as e:
                    print(f"⚠️ Error processing paper: {str(e)}")
                    continue
            
            print(f"✅ Successfully processed {len(results)} papers from Springer dataset")
            return results
            
        except Exception as e:
            print(f"❌ Error processing Springer JSON: {str(e)}")
            return []
    
    async def _process_pdf(self, file_path: str) -> Tuple[str, str, List[str], str]:
        """Extract content from PDF file"""
        try:
            content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            
            # Try to extract title (usually on first page, larger text)
            lines = content.split('\n')[:10]  # First 10 lines
            title = next((line.strip() for line in lines if len(line.strip()) > 10), "Untitled Document")
            
            # Basic metadata extraction (can be enhanced)
            authors = []  # PDF metadata parsing can be added here
            abstract = ""  # Abstract detection can be added here
            
            return content.strip(), title, authors, abstract
            
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")
    
    async def _process_json(self, file_path: str) -> Tuple[str, str, List[str], str]:
        """Process JSON file (like your Springer data)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle single paper JSON
            if isinstance(data, dict):
                title = data.get('title', 'Untitled')
                abstract = data.get('abstract', '')
                authors = data.get('authors', [])
                subjects = data.get('subjects', [])
                
                content = f"Title: {title}\n\nAbstract: {abstract}"
                if subjects:
                    content += f"\n\nSubjects: {', '.join(subjects)}"
                
                return content, title, authors, abstract
            
            # Handle multiple papers JSON
            elif isinstance(data, list) and len(data) > 0:
                # Process first paper as example
                paper = data[0]
                title = paper.get('title', 'Multi-paper Dataset')
                content = f"Dataset containing {len(data)} academic papers"
                return content, title, [], ""
            
            else:
                return str(data), "JSON Document", [], ""
                
        except Exception as e:
            raise ValueError(f"Error processing JSON: {str(e)}")
    
    async def _process_txt(self, file_path: str) -> Tuple[str, str, List[str], str]:
        """Process plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from first line
            lines = content.split('\n')
            title = lines[0].strip() if lines else "Text Document"
            
            return content, title, [], ""
            
        except Exception as e:
            raise ValueError(f"Error processing text file: {str(e)}")
    
    async def _create_chunks(
        self, 
        document_id: str, 
        content: str, 
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """
        Create overlapping text chunks for better retrieval.
        
        This is crucial for RAG - smaller chunks = more precise retrieval.
        """
        chunks = []
        
        # Clean the content
        content = self._clean_text(content)
        
        # Split into chunks with overlap
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                for i in range(min(100, self.chunk_size // 4)):
                    if content[end - i] in '.!?':
                        end = end - i + 1
                        break
            
            # Extract chunk content
            chunk_content = content[start:end].strip()
            
            if chunk_content:  # Only create non-empty chunks
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "document_title": metadata.title,
                        "document_source": metadata.source,
                        "chunk_length": len(chunk_content)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        return text.strip()
    
    def _generate_document_id(self, identifier: str, content: str) -> str:
        """Generate a unique document ID"""
        # Use MD5 hash of identifier + content snippet for uniqueness
        content_snippet = content[:500]  # First 500 chars
        combined = f"{identifier}_{content_snippet}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using sentence transformers.
        
        This is the key to semantic search - converting text to vectors.
        """
        try:
            # Run embedding generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.embedding_model.encode,
                text
            )
            return embedding.tolist()
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processing statistics"""
        processed_dir = Path(settings.PROCESSED_DIR)
        upload_dir = Path(settings.UPLOAD_DIR)
        
        return {
            "processed_files": len(list(processed_dir.glob("*"))) if processed_dir.exists() else 0,
            "uploaded_files": len(list(upload_dir.glob("*"))) if upload_dir.exists() else 0,
            "supported_formats": ["PDF", "JSON", "TXT"],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": settings.EMBEDDING_MODEL
        }
    
    

def process_single_springer_paper(paper_data):
    """
    Process one paper from your Springer dataset
    
    paper_data should have: source, identifier, title, abstract, subjects, etc.
    """
    
    # Create meaningful content for each paper
    paper_content = f"""
    Title: {paper_data.get('title', 'Unknown Title')}
    
    Abstract: {paper_data.get('abstract', 'No abstract available')}
    
    Subjects: {paper_data.get('subjects', 'No subjects listed')}
    
    Source: {paper_data.get('source', 'Springer')}
    Identifier: {paper_data.get('identifier', 'Unknown')}
    """
    
    # Create metadata for retrieval
    paper_metadata = {
        "document_title": paper_data.get('title'),
        "document_source": paper_data.get('source', 'Springer'),
        "identifier": paper_data.get('identifier'),
        "subjects": paper_data.get('subjects'),
        "document_type": "academic_paper"
    }
    
    return paper_content, paper_metadata

