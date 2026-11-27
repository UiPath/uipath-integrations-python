"""Web search integration for deep research using Tavily API."""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import aiohttp


@dataclass
class WebSearchResult:
    """Result from web search."""

    title: str
    content: str
    url: str
    score: float
    published_date: Optional[str] = None


class TavilyWebSearch:
    """Web search client using Tavily API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com"

        if not self.api_key:
            raise ValueError(
                "Tavily API key is required. Set TAVILY_API_KEY environment variable."
            )

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[WebSearchResult]:
        """
        Search the web using Tavily API.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_depth: "basic" or "advanced"
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
        """
        payload: Dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": True,
            "include_images": False,
            "include_raw_content": True,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:

                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Tavily API error {response.status}: {error_text}"
                        )

                    data = await response.json()
                    return self._parse_results(data)

        except Exception as e:
            # Return empty results on error rather than failing completely
            print(f"Web search error: {e}")
            return []

    def _parse_results(self, data: Dict[str, Any]) -> List[WebSearchResult]:
        """Parse Tavily API response into WebSearchResult objects."""
        results = []

        for item in data.get("results", []):
            result = WebSearchResult(
                title=item.get("title", ""),
                content=item.get("content", ""),
                url=item.get("url", ""),
                score=item.get("score", 0.0),
                published_date=item.get("published_date"),
            )
            results.append(result)

        return results


class MockWebSearch:
    """Mock web search for testing/demo purposes."""

    async def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> List[WebSearchResult]:
        """Mock search that returns sample results."""
        # Simulate API delay
        await asyncio.sleep(0.1)

        mock_results = []
        for i in range(min(max_results, 3)):
            result = WebSearchResult(
                title=f"Mock Result {i+1} for: {query}",
                content=f"""
                This is mock web search content for query: "{query}".

                In a real implementation, this would be actual web content
                retrieved from search engines. The content would provide
                relevant information about the query topic from various
                online sources including news articles, research papers,
                blog posts, and other web resources.

                Key points related to {query}:
                - Important finding {i+1}
                - Relevant data point {i+1}
                - Current trend analysis {i+1}
                """,
                url=f"https://example.com/result-{i+1}",
                score=0.9 - (i * 0.1),
                published_date="2024-01-15",
            )
            mock_results.append(result)

        return mock_results


def create_web_search_client(
    use_mock: bool = False,
) -> Union[TavilyWebSearch, MockWebSearch]:
    """Create web search client.

    Args:
        use_mock: If True, return mock client for testing
    """
    if use_mock or not os.getenv("TAVILY_API_KEY"):
        return MockWebSearch()
    else:
        return TavilyWebSearch()
