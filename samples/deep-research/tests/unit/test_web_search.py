"""
Unit tests for web search functionality
"""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from agents.web_search import (
    MockWebSearch,
    TavilyWebSearch,
    WebSearchResult,
    create_web_search_client,
)


class TestWebSearchResult:
    """Test WebSearchResult data class"""

    def test_web_search_result_creation(self):
        """Test creating WebSearchResult"""
        result = WebSearchResult(
            title="Test Title",
            content="Test content",
            url="https://example.com",
            score=0.9,
            published_date="2024-01-15",
        )

        assert result.title == "Test Title"
        assert result.content == "Test content"
        assert result.url == "https://example.com"
        assert result.score == 0.9
        assert result.published_date == "2024-01-15"

    def test_web_search_result_optional_date(self):
        """Test WebSearchResult with optional published_date"""
        result = WebSearchResult(
            title="Test", content="Content", url="https://example.com", score=0.8
        )

        assert result.published_date is None


class TestTavilyWebSearch:
    """Test TavilyWebSearch client"""

    def test_initialization_with_api_key(self):
        """Test TavilyWebSearch initialization with API key"""
        client = TavilyWebSearch(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.base_url == "https://api.tavily.com"

    def test_initialization_without_api_key(self):
        """Test TavilyWebSearch initialization without API key raises error"""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Tavily API key is required"):
                TavilyWebSearch()

    def test_initialization_with_env_var(self):
        """Test TavilyWebSearch initialization with environment variable"""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "env_key"}):
            client = TavilyWebSearch()
            assert client.api_key == "env_key"

    async def test_search_success(self):
        """Test successful search request"""
        mock_response_data = {
            "results": [
                {
                    "title": "Test Result 1",
                    "content": "Test content 1",
                    "url": "https://example1.com",
                    "score": 0.9,
                    "published_date": "2024-01-15",
                },
                {
                    "title": "Test Result 2",
                    "content": "Test content 2",
                    "url": "https://example2.com",
                    "score": 0.8,
                },
            ]
        }

        with patch("agents.web_search.aiohttp.ClientSession") as mock_session:
            # Mock the session and response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            # Create a proper async context manager mock
            from unittest.mock import MagicMock

            async def mock_aenter_session(self):
                return mock_session_instance

            async def mock_aexit_session(self, *args):
                return None

            async def mock_aenter_response(self):
                return mock_response

            async def mock_aexit_response(self, *args):
                return None

            mock_session_instance = MagicMock()
            mock_post_response = MagicMock()
            mock_post_response.__aenter__ = mock_aenter_response
            mock_post_response.__aexit__ = mock_aexit_response
            mock_session_instance.post.return_value = mock_post_response
            mock_session.return_value.__aenter__ = mock_aenter_session
            mock_session.return_value.__aexit__ = mock_aexit_session

            client = TavilyWebSearch(api_key="test_key")
            results = await client.search("test query")

            # Assert
            assert len(results) == 2
            assert results[0].title == "Test Result 1"
            assert results[0].content == "Test content 1"
            assert results[0].url == "https://example1.com"
            assert results[0].score == 0.9
            assert results[0].published_date == "2024-01-15"

            assert results[1].title == "Test Result 2"
            assert results[1].published_date is None

    async def test_search_with_parameters(self):
        """Test search with custom parameters"""
        mock_response_data = {"results": []}

        with patch("agents.web_search.aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            # Create a proper async context manager mock
            from unittest.mock import MagicMock

            async def mock_aenter_session(self):
                return mock_session_instance

            async def mock_aexit_session(self, *args):
                return None

            async def mock_aenter_response(self):
                return mock_response

            async def mock_aexit_response(self, *args):
                return None

            mock_session_instance = MagicMock()
            mock_post_response = MagicMock()
            mock_post_response.__aenter__ = mock_aenter_response
            mock_post_response.__aexit__ = mock_aexit_response
            mock_session_instance.post.return_value = mock_post_response
            mock_session.return_value.__aenter__ = mock_aenter_session
            mock_session.return_value.__aexit__ = mock_aexit_session

            client = TavilyWebSearch(api_key="test_key")
            await client.search(
                query="test query",
                max_results=10,
                search_depth="basic",
                include_domains=["example.com"],
                exclude_domains=["spam.com"],
            )

            # Verify the request was made with correct parameters
            mock_session_instance.post.assert_called_once()
            call_args = mock_session_instance.post.call_args
            request_data = call_args[1]["json"]

            assert request_data["query"] == "test query"
            assert request_data["max_results"] == 10
            assert request_data["search_depth"] == "basic"
            assert request_data["include_domains"] == ["example.com"]
            assert request_data["exclude_domains"] == ["spam.com"]

    async def test_search_api_error(self):
        """Test handling of API errors"""
        with patch("agents.web_search.aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 400
            mock_response.text = AsyncMock(return_value="Bad Request")

            # Create a proper async context manager mock
            from unittest.mock import MagicMock

            async def mock_aenter_session(self):
                return mock_session_instance

            async def mock_aexit_session(self, *args):
                return None

            async def mock_aenter_response(self):
                return mock_response

            async def mock_aexit_response(self, *args):
                return None

            mock_session_instance = MagicMock()
            mock_post_response = MagicMock()
            mock_post_response.__aenter__ = mock_aenter_response
            mock_post_response.__aexit__ = mock_aexit_response
            mock_session_instance.post.return_value = mock_post_response
            mock_session.return_value.__aenter__ = mock_aenter_session
            mock_session.return_value.__aexit__ = mock_aexit_session

            client = TavilyWebSearch(api_key="test_key")
            results = await client.search("test query")

            # Should return empty list on error
            assert results == []

    async def test_search_network_error(self):
        """Test handling of network errors"""
        with patch("agents.web_search.aiohttp.ClientSession") as mock_session:
            # Create a proper async context manager mock
            from unittest.mock import MagicMock

            async def mock_aenter_session(self):
                return mock_session_instance

            async def mock_aexit_session(self, *args):
                return None

            mock_session_instance = MagicMock()
            mock_session_instance.post.side_effect = aiohttp.ClientError(
                "Network error"
            )
            mock_session.return_value.__aenter__ = mock_aenter_session
            mock_session.return_value.__aexit__ = mock_aexit_session

            client = TavilyWebSearch(api_key="test_key")
            results = await client.search("test query")

            # Should return empty list on error
            assert results == []

    def test_parse_results_complete(self):
        """Test parsing complete API response"""
        client = TavilyWebSearch(api_key="test_key")

        api_data = {
            "results": [
                {
                    "title": "Complete Result",
                    "content": "Full content",
                    "url": "https://complete.com",
                    "score": 0.95,
                    "published_date": "2024-01-15",
                }
            ]
        }

        results = client._parse_results(api_data)

        assert len(results) == 1
        assert results[0].title == "Complete Result"
        assert results[0].content == "Full content"
        assert results[0].url == "https://complete.com"
        assert results[0].score == 0.95
        assert results[0].published_date == "2024-01-15"

    def test_parse_results_minimal(self):
        """Test parsing minimal API response"""
        client = TavilyWebSearch(api_key="test_key")

        api_data = {"results": [{"title": "Minimal Result"}]}

        results = client._parse_results(api_data)

        assert len(results) == 1
        assert results[0].title == "Minimal Result"
        assert results[0].content == ""
        assert results[0].url == ""
        assert results[0].score == 0.0
        assert results[0].published_date is None

    def test_parse_results_empty(self):
        """Test parsing empty API response"""
        client = TavilyWebSearch(api_key="test_key")

        api_data = {"results": []}
        results = client._parse_results(api_data)

        assert results == []


class TestMockWebSearch:
    """Test MockWebSearch client"""

    async def test_mock_search(self):
        """Test mock search functionality"""
        client = MockWebSearch()
        results = await client.search("test query", max_results=2)

        assert len(results) == 2
        assert all(isinstance(result, WebSearchResult) for result in results)
        assert "test query" in results[0].title
        assert "test query" in results[0].content
        assert results[0].url.startswith("https://example.com/")
        assert results[0].score > results[1].score  # Scores should decrease

    async def test_mock_search_max_results(self):
        """Test mock search respects max_results"""
        client = MockWebSearch()

        # Request more than available
        results = await client.search("query", max_results=10)
        assert len(results) == 3  # Mock returns max 3

        # Request fewer
        results = await client.search("query", max_results=1)
        assert len(results) == 1

    async def test_mock_search_delay(self):
        """Test mock search includes simulated delay"""
        import time

        client = MockWebSearch()
        start_time = time.time()
        await client.search("test")
        elapsed = time.time() - start_time

        # Should have some delay (at least 0.1 seconds)
        assert elapsed >= 0.1


class TestCreateWebSearchClient:
    """Test web search client factory function"""

    def test_create_mock_client(self):
        """Test creating mock client"""
        client = create_web_search_client(use_mock=True)
        assert isinstance(client, MockWebSearch)

    def test_create_real_client_with_api_key(self):
        """Test creating real client when API key is available"""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            client = create_web_search_client(use_mock=False)
            assert isinstance(client, TavilyWebSearch)
            assert client.api_key == "test_key"

    def test_create_mock_when_no_api_key(self):
        """Test creating mock client when no API key is available"""
        with patch.dict("os.environ", {}, clear=True):
            client = create_web_search_client(use_mock=False)
            assert isinstance(client, MockWebSearch)
