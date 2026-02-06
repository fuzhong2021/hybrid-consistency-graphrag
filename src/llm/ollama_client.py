# src/llm/ollama_client.py
"""
Ollama-Client mit OpenAI-kompatiblem Interface.

Ermöglicht den Einsatz lokaler LLMs (Llama, Mistral, etc.) als
Drop-in-Replacement für OpenAI.

Verwendung:
    from src.llm.ollama_client import OllamaClient

    client = OllamaClient(model="llama3.1:8b")

    # OpenAI-kompatibles Interface
    response = client.chat.completions.create(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.0
    )
"""

import json
import logging
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    """OpenAI-kompatible Response-Struktur."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ChatCompletions:
    """Wrapper für chat.completions.create()"""

    def __init__(self, client: 'OllamaClient'):
        self.client = client

    def create(
        self,
        model: str = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Dict = None,
        **kwargs
    ) -> ChatCompletion:
        """
        OpenAI-kompatible chat completion.

        Args:
            model: Modellname (z.B. "llama3.1:8b")
            messages: Liste von Messages
            temperature: Sampling-Temperatur
            max_tokens: Maximale Ausgabe-Tokens
            response_format: {"type": "json_object"} für JSON-Modus

        Returns:
            ChatCompletion-Objekt
        """
        model = model or self.client.model

        # Request an Ollama API
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        # JSON-Modus
        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.client.base_url}/api/chat",
                json=payload,
                timeout=300  # 5 Minuten Timeout für lange Generierungen
            )
            response.raise_for_status()
            data = response.json()

            # Zu OpenAI-Format konvertieren
            return ChatCompletion(
                id=f"ollama-{data.get('created_at', '')}",
                object="chat.completion",
                created=0,
                model=model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=data.get("message", {}).get("content", "")
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                )
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API Fehler: {e}")
            raise


class Chat:
    """Wrapper für client.chat"""

    def __init__(self, client: 'OllamaClient'):
        self.completions = ChatCompletions(client)


class OllamaClient:
    """
    Ollama-Client mit OpenAI-kompatiblem Interface.

    Unterstützt lokale LLMs wie Llama, Mistral, Qwen, etc.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Args:
            model: Standard-Modell (z.B. "llama3.1:8b", "mistral:latest")
            base_url: Ollama API URL
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chat = Chat(self)

        # Verbindung prüfen
        self._check_connection()
        logger.info(f"OllamaClient initialisiert: {model} @ {base_url}")

    def _check_connection(self):
        """Prüft ob Ollama erreichbar ist."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Ollama nicht erreichbar unter {self.base_url}. "
                "Starte Ollama mit: ollama serve"
            ) from e

    def list_models(self) -> List[str]:
        """Listet verfügbare Modelle."""
        response = requests.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]

    def pull_model(self, model: str) -> bool:
        """Lädt ein Modell herunter."""
        logger.info(f"Lade Modell: {model}")
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": model},
            stream=True
        )
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "status" in data:
                    logger.info(data["status"])
        return True


def create_llm_client(
    provider: str = "ollama",
    model: str = None,
    api_key: str = None,
    **kwargs
):
    """
    Factory-Funktion für LLM-Clients.

    Args:
        provider: "ollama" oder "openai"
        model: Modellname
        api_key: API-Key (nur für OpenAI)

    Returns:
        LLM-Client mit OpenAI-kompatiblem Interface
    """
    if provider == "ollama":
        model = model or "llama3.1:8b"
        return OllamaClient(model=model, **kwargs)

    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    else:
        raise ValueError(f"Unbekannter Provider: {provider}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Ollama Client Test ===\n")

    # Client erstellen
    client = OllamaClient(model="llama3.1:8b")

    print(f"Verfügbare Modelle: {client.list_models()}\n")

    # Einfacher Test
    print("Test: Einfache Frage...")
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Was ist 2+2? Antworte nur mit der Zahl."}
        ],
        temperature=0.0
    )
    print(f"Antwort: {response.choices[0].message.content}")

    # JSON-Modus Test
    print("\nTest: JSON-Extraktion...")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Extrahiere Entitäten als JSON."},
            {"role": "user", "content": "Albert Einstein wurde in Ulm geboren."}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    print(f"JSON: {response.choices[0].message.content}")

    print("\n=== Test abgeschlossen ===")
