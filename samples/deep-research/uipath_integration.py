"""UiPath Context Grounding Integration for Deep Research Agent.

This module provides integration with UiPath Context Grounding services,
implementing real query engines for enterprise data retrieval.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp  # type: ignore
from llama_index.core.base.response.schema import Response  # type: ignore
from llama_index.core.llms import LLM  # type: ignore
from llama_index.core.query_engine import BaseQueryEngine  # type: ignore
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode  # type: ignore


class UiPathContextGroundingQueryEngine(BaseQueryEngine):
    """Query engine that integrates with UiPath Context Grounding service.

    This engine connects to UiPath's Context Grounding API to retrieve
    relevant context from enterprise data sources.
    """

    def __init__(
        self,
        orchestrator_url: str,
        tenant_name: str,
        client_id: str,
        client_secret: str,
        index_name: str,
        context_type: str = "general",
        max_results: int = 5,
        similarity_threshold: float = 0.7,
        llm: Optional[LLM] = None,
    ):
        """Initialize UiPath Context Grounding Query Engine.

        Args:
            orchestrator_url: UiPath Orchestrator URL
            tenant_name: UiPath tenant name
            client_id: OAuth client ID
            client_secret: OAuth client secret
            index_name: Context grounding index name
            context_type: Type of context (e.g., "policy", "technical", "knowledge")
            max_results: Maximum number of results to retrieve
            similarity_threshold: Minimum similarity score for results
            llm: LLM instance for response synthesis
        """
        super().__init__(callback_manager=None)
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.tenant_name = tenant_name
        self.client_id = client_id
        self.client_secret = client_secret
        self.index_name = index_name
        self.context_type = context_type
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
        self.llm = llm

        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

    async def _get_access_token(self) -> str:
        """Get OAuth access token for UiPath API."""
        # Check if current token is still valid
        if (
            self._access_token
            and self._token_expires_at
            and datetime.now() < self._token_expires_at
        ):
            return self._access_token

        # Request new token
        token_url = f"{self.orchestrator_url}/api/account/authenticate"

        async with aiohttp.ClientSession() as session:
            payload = {
                "tenancyName": self.tenant_name,
                "usernameOrEmailAddress": self.client_id,
                "password": self.client_secret,
                "grantType": "client_credentials",
            }

            try:
                async with session.post(token_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._access_token = data.get("result")
                        # Assume token expires in 1 hour
                        self._token_expires_at = datetime.now().replace(
                            hour=datetime.now().hour + 1
                        )
                        return self._access_token
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Authentication failed: {response.status} - {error_text}"
                        )

            except Exception as e:
                raise Exception(f"Failed to authenticate with UiPath: {str(e)}")

    async def _query_context_grounding(self, query: str) -> List[Dict[str, Any]]:
        """Query the UiPath Context Grounding service."""
        access_token = await self._get_access_token()

        # Context Grounding API endpoint
        api_url = f"{self.orchestrator_url}/api/contextgrounding/v1/search"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-UIPATH-TenantName": self.tenant_name,
        }

        payload = {
            "query": query,
            "indexName": self.index_name,
            "maxResults": self.max_results,
            "similarityThreshold": self.similarity_threshold,
            "includeMetadata": True,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    api_url, json=payload, headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Context grounding query failed: {response.status} - {error_text}"
                        )

            except Exception as e:
                raise Exception(f"Failed to query context grounding: {str(e)}")

    def _create_nodes_from_results(
        self, results: List[Dict[str, Any]]
    ) -> List[NodeWithScore]:
        """Convert context grounding results to LlamaIndex nodes."""
        nodes = []

        for result in results:
            content = result.get("content", "")
            score = result.get("score", 0.0)
            metadata = result.get("metadata", {})

            # Enhance metadata with context type and source info
            enhanced_metadata = {
                "source": f"UiPath {self.context_type}",
                "context_type": self.context_type,
                "index_name": self.index_name,
                "similarity_score": score,
                **metadata,
            }

            node = TextNode(text=content, metadata=enhanced_metadata)

            nodes.append(NodeWithScore(node=node, score=score))

        return nodes

    async def aquery(self, query_bundle: QueryBundle) -> Response:
        """Execute async query against UiPath Context Grounding."""
        query_str = query_bundle.query_str

        try:
            # Query context grounding service
            results = await self._query_context_grounding(query_str)

            # Convert to nodes
            nodes = self._create_nodes_from_results(results)

            # Generate response text
            if nodes:
                # Combine content from all nodes
                combined_content = "\n\n".join(
                    [
                        f"[Source: {node.node.metadata.get('source', 'Unknown')}]\n{getattr(node.node, 'text', str(node.node))}"
                        for node in nodes
                    ]
                )

                if self.llm:
                    # Use LLM to synthesize response
                    synthesis_prompt = f"""
                    Based on the following context information, provide a comprehensive answer to the query: "{query_str}"

                    Context:
                    {combined_content}

                    Please provide a clear, accurate response based on the context provided.
                    """

                    llm_response = await self.llm.acomplete(synthesis_prompt)
                    response_text = str(llm_response)
                else:
                    # Return combined content directly
                    response_text = combined_content
            else:
                response_text = f"No relevant information found in {self.context_type} for query: {query_str}"

            return Response(
                response=response_text,
                source_nodes=nodes,
                metadata={
                    "context_type": self.context_type,
                    "index_name": self.index_name,
                    "num_results": len(nodes),
                },
            )

        except Exception as e:
            error_response = f"Error querying {self.context_type} context: {str(e)}"
            return Response(
                response=error_response,
                source_nodes=[],
                metadata={"error": str(e), "context_type": self.context_type},
            )

    def query(self, query_bundle: QueryBundle) -> Response:
        """Execute synchronous query interface."""
        return asyncio.run(self.aquery(query_bundle))

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Implements abstract _query method."""
        return self.query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Implements abstract _aquery method."""
        return await self.aquery(query_bundle)

    def _get_prompt_modules(self) -> Dict[str, Any]:
        """Implements abstract _get_prompt_modules method."""
        return {}


def create_uipath_query_engines(
    orchestrator_url: str,
    tenant_name: str,
    client_id: str,
    client_secret: str,
    llm: Optional[LLM] = None,
) -> Dict[str, UiPathContextGroundingQueryEngine]:
    """Create UiPath Context Grounding query engines for different data types.

    Args:
        orchestrator_url: UiPath Orchestrator URL
        tenant_name: UiPath tenant name
        client_id: OAuth client ID
        client_secret: OAuth client secret
        llm: LLM instance for response synthesis

    Returns:
        Dictionary of query engines for different context types
    """
    engines = {}

    # Define different context types and their corresponding indexes
    context_configs = [
        {
            "name": "company_policy",
            "index": "company-policies-index",
            "description": "Company policies and procedures",
        },
        {
            "name": "technical_docs",
            "index": "technical-documentation-index",
            "description": "Technical documentation and guides",
        },
        {
            "name": "knowledge_base",
            "index": "knowledge-base-index",
            "description": "General knowledge base articles",
        },
        {
            "name": "compliance",
            "index": "compliance-guidelines-index",
            "description": "Compliance and regulatory guidelines",
        },
        {
            "name": "best_practices",
            "index": "best-practices-index",
            "description": "Industry best practices and standards",
        },
    ]

    for config in context_configs:
        engines[config["name"]] = UiPathContextGroundingQueryEngine(
            orchestrator_url=orchestrator_url,
            tenant_name=tenant_name,
            client_id=client_id,
            client_secret=client_secret,
            index_name=config["index"],
            context_type=config["name"],
            llm=llm,
        )

    return engines


def validate_uipath_config(config: Dict[str, Any]) -> bool:
    """Validate UiPath configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid, False otherwise
    """
    uipath_config = config.get("uipath", {})

    required_fields = ["orchestrator_url", "tenant_name", "client_id", "client_secret"]

    for field in required_fields:
        if not uipath_config.get(field):
            return False

    return True


async def test_uipath_connection(
    orchestrator_url: str, tenant_name: str, client_id: str, client_secret: str
) -> Dict[str, Any]:
    """Test connection to UiPath services.

    Returns:
        Dictionary with connection test results
    """
    result = {"success": False, "message": "", "details": {}}

    try:
        # Create a test query engine
        test_engine = UiPathContextGroundingQueryEngine(
            orchestrator_url=orchestrator_url,
            tenant_name=tenant_name,
            client_id=client_id,
            client_secret=client_secret,
            index_name="test-index",
            context_type="test",
        )

        # Test authentication
        token = await test_engine._get_access_token()

        if token:
            result["success"] = True  # type: ignore
            result["message"] = "Successfully connected to UiPath services"  # type: ignore
            result["details"]["authenticated"] = True  # type: ignore
            result["details"]["token_length"] = len(token)  # type: ignore
        else:
            result["message"] = "Authentication failed - no token received"  # type: ignore

    except Exception as e:
        result["message"] = f"Connection test failed: {str(e)}"  # type: ignore
        result["details"]["error"] = str(e)  # type: ignore

    return result
