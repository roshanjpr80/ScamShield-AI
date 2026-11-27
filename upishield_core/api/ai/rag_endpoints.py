"""
RAG API Endpoints for Fraud Knowledge Base.

Provides REST endpoints for searching and managing the fraud knowledge base.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from upishield_core.ai import (
    FraudKnowledgeBase,
    KnowledgeCategory,
    KnowledgeChunk,
    SearchResult,
    get_knowledge_base,
)

router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base (RAG)"])


# Pydantic Models

class KnowledgeSearchRequest(BaseModel):
    """Request for knowledge base search."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)
    category: Optional[KnowledgeCategory] = Field(None, description="Filter by category")
    min_score: float = Field(0.0, description="Minimum similarity score", ge=0.0, le=1.0)


class KnowledgeSearchResponse(BaseModel):
    """Response for knowledge base search."""
    query: str
    results: List[Dict[str, Any]]
    total_found: int
    search_time_ms: float


class KnowledgeChunkRequest(BaseModel):
    """Request to add a knowledge chunk."""
    title: str = Field(..., description="Chunk title", min_length=1)
    content: str = Field(..., description="Chunk content", min_length=10)
    category: KnowledgeCategory = Field(..., description="Knowledge category")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering")
    source: str = Field("user", description="Source of the knowledge")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KnowledgeChunkResponse(BaseModel):
    """Response for knowledge chunk operations."""
    chunk: Dict[str, Any]
    operation: str
    success: bool


class KnowledgeStatsResponse(BaseModel):
    """Knowledge base statistics."""
    total_chunks: int
    categories: Dict[str, int]
    embedding_model: str
    embedding_dimension: int
    initialized: bool


class KnowledgeRebuildResponse(BaseModel):
    """Response for index rebuild."""
    success: bool
    chunks_reindexed: int
    message: str


# Dependency

def get_kb() -> FraudKnowledgeBase:
    """Get knowledge base instance."""
    return get_knowledge_base()


# Endpoints

@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    request: KnowledgeSearchRequest,
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """
    Search the fraud knowledge base.

    Uses semantic search with sentence-transformers embeddings
    to find relevant fraud patterns, regulations, case studies, and prevention tips.
    """
    import time
    start = time.time()

    try:
        results = kb.search(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category,
            min_score=request.min_score,
        )

        search_time_ms = (time.time() - start) * 1000

        return KnowledgeSearchResponse(
            query=request.query,
            results=[r.to_dict() for r in results],
            total_found=len(results),
            search_time_ms=search_time_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge_get(
    q: str = Query(..., description="Search query", min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    category: Optional[KnowledgeCategory] = Query(None),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Search the knowledge base via GET request."""
    return await search_knowledge(
        KnowledgeSearchRequest(
            query=q,
            top_k=top_k,
            category=category,
            min_score=min_score,
        ),
        kb,
    )


@router.get("/chunk/{chunk_id}", response_model=KnowledgeChunkResponse)
async def get_chunk(
    chunk_id: str,
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Get a specific knowledge chunk by ID."""
    chunk = kb.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return KnowledgeChunkResponse(
        chunk=chunk.to_dict(),
        operation="get",
        success=True,
    )


@router.post("/chunk", response_model=KnowledgeChunkResponse)
async def add_chunk(
    request: KnowledgeChunkRequest,
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Add a new knowledge chunk to the knowledge base."""
    try:
        chunk = KnowledgeChunk(
            id=str(uuid.uuid4())[:8],
            title=request.title,
            content=request.content,
            category=request.category,
            tags=request.tags,
            source=request.source,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=request.metadata,
        )

        kb.add_chunk(chunk)

        return KnowledgeChunkResponse(
            chunk=chunk.to_dict(),
            operation="create",
            success=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add chunk: {str(e)}")


@router.post("/chunks/bulk", response_model=KnowledgeChunkResponse)
async def add_chunks_bulk(
    requests: List[KnowledgeChunkRequest] = Body(...),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Add multiple knowledge chunks in bulk."""
    if not requests:
        raise HTTPException(status_code=400, detail="No chunks provided")

    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 chunks per request")

    try:
        chunks = []
        for req in requests:
            chunks.append(KnowledgeChunk(
                id=str(uuid.uuid4())[:8],
                title=req.title,
                content=req.content,
                category=req.category,
                tags=req.tags,
                source=req.source,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=req.metadata,
            ))

        kb.add_chunks(chunks)

        return KnowledgeChunkResponse(
            chunk={"added": len(chunks), "ids": [c.id for c in chunks]},
            operation="bulk_create",
            success=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk add failed: {str(e)}")


@router.delete("/chunk/{chunk_id}")
async def delete_chunk(
    chunk_id: str,
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Delete a knowledge chunk by ID."""
    # Note: VectorStore doesn't expose individual delete easily
    # This would require extending the VectorStore class
    raise HTTPException(status_code=501, detail="Delete not yet implemented")


@router.get("/stats", response_model=KnowledgeStatsResponse)
async def get_stats(
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Get knowledge base statistics."""
    stats = kb.get_stats()
    return KnowledgeStatsResponse(**stats)


@router.post("/rebuild", response_model=KnowledgeRebuildResponse)
async def rebuild_index(
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Rebuild the vector index from stored chunks."""
    try:
        chunks_before = kb.vector_store.count()
        kb.rebuild_index()
        chunks_after = kb.vector_store.count()

        return KnowledgeRebuildResponse(
            success=True,
            chunks_reindexed=chunks_after,
            message=f"Rebuilt index with {chunks_after} chunks",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@router.post("/initialize")
async def initialize_kb(
    seed_data: bool = Query(True, description="Load seed fraud knowledge"),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Initialize the knowledge base with optional seed data."""
    try:
        success = kb.initialize(seed_data=seed_data)
        return {
            "success": success,
            "initialized": kb._initialized,
            "chunks_loaded": kb.vector_store.count() if success else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.get("/categories")
async def get_categories():
    """Get all available knowledge categories."""
    return {
        "categories": [{"value": c.value, "name": c.name} for c in KnowledgeCategory]
    }


# ============================================================
# Specialized Search Endpoints
# ============================================================

@router.post("/search/upi-fraud")
async def search_upi_fraud(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Search specifically for UPI fraud patterns."""
    results = kb.search(
        query=query,
        top_k=top_k,
        category_filter=KnowledgeCategory.UPI_FRAUD,
    )
    return {"query": query, "results": [r.to_dict() for r in results]}


@router.post("/search/sms-phishing")
async def search_sms_phishing(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Search specifically for SMS phishing patterns."""
    results = kb.search(
        query=query,
        top_k=top_k,
        category_filter=KnowledgeCategory.SMS_PHISHING,
    )
    return {"query": query, "results": [r.to_dict() for r in results]}


@router.post("/search/job-scams")
async def search_job_scams(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Search specifically for job scam patterns."""
    results = kb.search(
        query=query,
        top_k=top_k,
        category_filter=KnowledgeCategory.JOB_SCAM,
    )
    return {"query": query, "results": [r.to_dict() for r in results]}


@router.post("/search/prevention")
async def search_prevention(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Search for fraud prevention tips."""
    results = kb.search(
        query=query,
        top_k=top_k,
        category_filter=KnowledgeCategory.PREVENTION,
    )
    return {"query": query, "results": [r.to_dict() for r in results]}


@router.post("/search/regulations")
async def search_regulations(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Search for fraud-related regulations."""
    results = kb.search(
        query=query,
        top_k=top_k,
        category_filter=KnowledgeCategory.REGULATIONS,
    )
    return {"query": query, "results": [r.to_dict() for r in results]}


@router.post("/search/iocs")
async def search_iocs(
    query: str = Body(..., embed=True),
    top_k: int = Body(10, embed=True),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Search for Indicators of Compromise."""
    results = kb.search(
        query=query,
        top_k=top_k,
        category_filter=KnowledgeCategory.IOC,
    )
    return {"query": query, "results": [r.to_dict() for r in results]}


@router.post("/search/case-studies")
async def search_case_studies(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """Search for fraud case studies."""
    results = kb.search(
        query=query,
        top_k=top_k,
        category_filter=KnowledgeCategory.CASE_STUDIES,
    )
    return {"query": query, "results": [r.to_dict() for r in results]}


# ============================================================
# RAG Integration for Agentic Workflow
# ============================================================

class RAGQueryRequest(BaseModel):
    """RAG query for agentic workflow integration."""
    query: str
    context_type: str = "general"  # "investigation", "report", "pattern", "prevention"
    top_k: int = 5


class RAGQueryResponse(BaseModel):
    """RAG query response formatted for agent consumption."""
    query: str
    context: str  # Formatted context for LLM
    sources: List[Dict[str, Any]]
    confidence: float


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    kb: FraudKnowledgeBase = Depends(get_kb),
):
    """
    RAG query endpoint for agentic workflow integration.

    Returns formatted context suitable for LLM consumption with citations.
    """
    # Determine category based on context_type
    category_map = {
        "investigation": KnowledgeCategory.CASE_STUDIES,
        "report": KnowledgeCategory.REGULATIONS,
        "pattern": KnowledgeCategory.UPI_FRAUD,
        "prevention": KnowledgeCategory.PREVENTION,
        "ioc": KnowledgeCategory.IOC,
    }

    category = category_map.get(request.context_type)

    results = kb.search(
        query=request.query,
        top_k=request.top_k,
        category_filter=category,
        min_score=0.3,
    )

    # Format context for LLM
    if not results:
        context = "No relevant knowledge found for this query."
        confidence = 0.0
    else:
        context_parts = []
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            context_parts.append(
                f"[Source {i}] {chunk.title} (Category: {chunk.category.value}, "
                f"Score: {result.score:.2f})\n{chunk.content}\n"
            )
        context = "\n---\n".join(context_parts)
        confidence = sum(r.score for r in results) / len(results)

    return RAGQueryResponse(
        query=request.query,
        context=context,
        sources=[r.to_dict() for r in results],
        confidence=confidence,
    )