"""
LLM Client abstraction for multiple providers.
Supports OpenAI, Anthropic, and local models via Ollama/vLLM.
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"
    AZURE_OPENAI = "azure_openai"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: Optional[List[Dict]] = None
    usage: Optional[Dict[str, int]] = None
    model: str = ""
    finish_reason: str = "stop"


@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    # Azure-specific
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        pass

    @abstractmethod
    async def close(self):
        pass

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        client = self._get_client()

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        if self.config.base_url:
            base_url = self.config.base_url.rstrip("/")
        else:
            base_url = "https://api.openai.com/v1"

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    **({"tool_calls": m.tool_calls} if m.tool_calls else {}),
                    **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
                    **({"name": m.name} if m.name else {}),
                }
                for m in messages
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        if stream:
            return self._stream_response(client, base_url, headers, payload)
        else:
            return await self._single_response(client, base_url, headers, payload)

    async def _single_response(
        self, client: httpx.AsyncClient, base_url: str, headers: Dict, payload: Dict
    ) -> LLMResponse:
        response = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = None
        if "tool_calls" in message:
            tool_calls = [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in message["tool_calls"]
            ]

        return LLMResponse(
            content=message.get("content", "") or "",
            tool_calls=tool_calls,
            usage=data.get("usage"),
            model=data.get("model", self.config.model),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def _stream_response(
        self, client: httpx.AsyncClient, base_url: str, headers: Dict, payload: Dict
    ) -> AsyncGenerator[str, None]:
        async with client.stream("POST", f"{base_url}/chat/completions", headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        client = self._get_client()

        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        base_url = self.config.base_url or "https://api.anthropic.com/v1"

        # Convert messages to Anthropic format
        system_prompt = ""
        anthropic_messages = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system_prompt = m.content
            else:
                anthropic_messages.append({"role": m.role.value, "content": m.content})

        payload = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = self._convert_tools(tools)
            if tool_choice:
                payload["tool_choice"] = self._convert_tool_choice(tool_choice)

        if stream:
            return self._stream_response(client, base_url, headers, payload)
        else:
            return await self._single_response(client, base_url, headers, payload)

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tools to Anthropic format."""
        converted = []
        for tool in tools:
            if tool["type"] == "function":
                fn = tool["function"]
                converted.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        return converted

    def _convert_tool_choice(self, tool_choice: Union[str, Dict]) -> Dict:
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                return {"type": "none"}
            else:
                return {"type": "tool", "name": tool_choice}
        elif isinstance(tool_choice, dict) and "function" in tool_choice:
            return {"type": "tool", "name": tool_choice["function"]["name"]}
        return {"type": "auto"}

    async def _single_response(
        self, client: httpx.AsyncClient, base_url: str, headers: Dict, payload: Dict
    ) -> LLMResponse:
        response = await client.post(f"{base_url}/messages", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        content = ""
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    },
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=data.get("usage"),
            model=data.get("model", self.config.model),
            finish_reason=data.get("stop_reason", "stop"),
        )

    async def _stream_response(
        self, client: httpx.AsyncClient, base_url: str, headers: Dict, payload: Dict
    ) -> AsyncGenerator[str, None]:
        async with client.stream("POST", f"{base_url}/messages", headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                yield delta["text"]
                    except json.JSONDecodeError:
                        continue

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client."""

    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        client = self._get_client()

        base_url = self.config.base_url or "http://localhost:11434"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools

        if stream:
            return self._stream_response(client, base_url, payload)
        else:
            return await self._single_response(client, base_url, payload)

    async def _single_response(
        self, client: httpx.AsyncClient, base_url: str, payload: Dict
    ) -> LLMResponse:
        response = await client.post(f"{base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        return LLMResponse(
            content=message.get("content", ""),
            tool_calls=None,  # Ollama doesn't support tools in same way
            model=data.get("model", self.config.model),
            finish_reason="stop" if data.get("done") else "length",
        )

    async def _stream_response(
        self, client: httpx.AsyncClient, base_url: str, payload: Dict
    ) -> AsyncGenerator[str, None]:
        async with client.stream("POST", f"{base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                try:
                    data = json.loads(line)
                    message = data.get("message", {})
                    if "content" in message and message["content"]:
                        yield message["content"]
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class VLLMClient(OpenAIClient):
    """vLLM client (OpenAI-compatible)."""

    def __init__(self, config: LLMConfig):
        # vLLM uses OpenAI-compatible API
        if not config.base_url:
            config.base_url = "http://localhost:8000/v1"
        if not config.api_key:
            config.api_key = "EMPTY"  # vLLM doesn't require real key
        super().__init__(config)


class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI client."""

    def __init__(self, config: LLMConfig):
        if not config.api_version:
            config.api_version = "2024-02-15-preview"
        if not config.deployment_name:
            config.deployment_name = config.model
        super().__init__(config)

    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        client = self._get_client()

        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json",
        }

        base_url = self.config.base_url.rstrip("/")
        deployment = self.config.deployment_name

        payload = {
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    **({"tool_calls": m.tool_calls} if m.tool_calls else {}),
                    **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
                    **({"name": m.name} if m.name else {}),
                }
                for m in messages
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        url = f"{base_url}/openai/deployments/{deployment}/chat/completions?api-version={self.config.api_version}"

        if stream:
            return self._stream_response(client, base_url, headers, payload)
        else:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            message = choice["message"]

            tool_calls = None
            if "tool_calls" in message:
                tool_calls = [
                    {
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in message["tool_calls"]
                ]

            return LLMResponse(
                content=message.get("content", "") or "",
                tool_calls=tool_calls,
                usage=data.get("usage"),
                model=data.get("model", self.config.model),
                finish_reason=choice.get("finish_reason", "stop"),
            )

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Factory function to create LLM client based on provider."""
    clients = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.OLLAMA: OllamaClient,
        LLMProvider.VLLM: VLLMClient,
        LLMProvider.AZURE_OPENAI: AzureOpenAIClient,
    }

    client_class = clients.get(config.provider)
    if not client_class:
        raise ValueError(f"Unsupported provider: {config.provider}")

    return client_class(config)


def get_default_llm_config() -> LLMConfig:
    """Get default LLM configuration from environment."""
    provider_str = os.getenv("LLM_PROVIDER", "openai").lower()
    provider = LLMProvider(provider_str)

    return LLMConfig(
        provider=provider,
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
        api_version=os.getenv("LLM_API_VERSION"),
        deployment_name=os.getenv("LLM_DEPLOYMENT_NAME"),
    )


class LLMManager:
    """Manager for LLM clients with fallback support."""

    def __init__(self, configs: Optional[List[LLMConfig]] = None):
        self.clients: List[BaseLLMClient] = []
        self.current_index = 0

        if configs:
            for config in configs:
                self.clients.append(create_llm_client(config))
        else:
            # Try to create default client
            try:
                config = get_default_llm_config()
                if config.api_key or config.provider in (LLMProvider.OLLAMA, LLMProvider.VLLM):
                    self.clients.append(create_llm_client(config))
            except Exception:
                pass

    def add_client(self, config: LLMConfig):
        self.clients.append(create_llm_client(config))

    async def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Try clients in order until one succeeds."""
        last_error = None

        for i, client in enumerate(self.clients):
            try:
                self.current_index = i
                return await client.chat(messages, tools, tool_choice, stream)
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"All LLM clients failed. Last error: {last_error}")

    async def close_all(self):
        for client in self.clients:
            await client.close()

    @property
    def active_client(self) -> Optional[BaseLLMClient]:
        if self.clients:
            return self.clients[self.current_index]
        return None