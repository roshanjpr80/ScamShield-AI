"""
AI Module for ScamShield - LLM Integration, Agentic Workflows, RAG, and Prompts.
"""

from upishield_core.ai.llm_client import (
    BaseLLMClient,
    LLMConfig,
    LLMManager,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    create_llm_client,
    get_default_llm_config,
)

from upishield_core.ai.prompts import (
    AI_ASSISTANT_PROMPT,
    DECISION_CONSENSUS_PROMPT,
    FEATURE_EXTRACTION_PROMPT,
    FRAUD_TOOLS,
    INVESTIGATION_PROMPT,
    PATTERN_RECOGNITION_PROMPT,
    REPORT_GENERATION_PROMPT,
    SYSTEM_PROMPT_DECISION_CONSENSUS,
    SYSTEM_PROMPT_FEATURE_EXTRACTION,
    SYSTEM_PROMPT_INVESTIGATION,
    SYSTEM_PROMPT_PATTERN_RECOGNITION,
    SYSTEM_PROMPT_REPORT_GENERATION,
    format_prompt,
    get_agent_prompt,
    get_tools_for_agent,
)

from upishield_core.ai.agentic_workflow import (
    AgentState,
    Decision,
    FraudDetectionResult,
    FraudDetectionWorkflow,
    FraudFeatures,
    FraudInputType,
    FraudReport,
    ModelPrediction,
    PatternMatch,
    RiskLevel,
    RuleResult,
    ThreatIntelResult,
    run_agentic_fraud_analysis,
)

from upishield_core.ai.knowledge_base import (
    FraudKnowledgeBase,
    KnowledgeCategory,
    KnowledgeChunk,
    SearchResult,
    VectorStore,
    EmbeddingProvider,
    get_knowledge_base,
)

from upishield_core.ai.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider as EmbeddingProviderBase,
    SentenceTransformerProvider,
    HFInferenceProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
    embed_texts_async,
    cosine_similarity,
    batch_cosine_similarity,
    get_top_k_similar,
)

__all__ = [
    # LLM Client
    "BaseLLMClient",
    "LLMConfig",
    "LLMManager",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "create_llm_client",
    "get_default_llm_config",
    # Prompts
    "AI_ASSISTANT_PROMPT",
    "DECISION_CONSENSUS_PROMPT",
    "FEATURE_EXTRACTION_PROMPT",
    "FRAUD_TOOLS",
    "INVESTIGATION_PROMPT",
    "PATTERN_RECOGNITION_PROMPT",
    "REPORT_GENERATION_PROMPT",
    "SYSTEM_PROMPT_DECISION_CONSENSUS",
    "SYSTEM_PROMPT_FEATURE_EXTRACTION",
    "SYSTEM_PROMPT_INVESTIGATION",
    "SYSTEM_PROMPT_PATTERN_RECOGNITION",
    "SYSTEM_PROMPT_REPORT_GENERATION",
    "format_prompt",
    "get_agent_prompt",
    "get_tools_for_agent",
    # Agentic Workflow
    "AgentState",
    "Decision",
    "FraudDetectionResult",
    "FraudDetectionWorkflow",
    "FraudFeatures",
    "FraudInputType",
    "FraudReport",
    "ModelPrediction",
    "PatternMatch",
    "RiskLevel",
    "RuleResult",
    "ThreatIntelResult",
    "run_agentic_fraud_analysis",
    # Knowledge Base (RAG)
    "FraudKnowledgeBase",
    "KnowledgeCategory",
    "KnowledgeChunk",
    "SearchResult",
    "VectorStore",
    "EmbeddingProvider",
    "get_knowledge_base",
    # Embeddings
    "EmbeddingConfig",
    "EmbeddingProviderBase",
    "SentenceTransformerProvider",
    "HFInferenceProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "create_embedding_provider",
    "embed_texts_async",
    "cosine_similarity",
    "batch_cosine_similarity",
    "get_top_k_similar",
]