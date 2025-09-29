import os
import json
import requests
from typing import Any, List, Optional, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr


class ChatBedrockWithApiKey(BaseChatModel):
    """
    Custom LangChain chat model to invoke Amazon Bedrock using a bearer token.
    """
    model_id: str
    api_key: SecretStr
    region: str = "ap-southeast-1"
    model_kwargs: Optional[dict] = None

    @property
    def _llm_type(self) -> str:
        return "bedrock-custom-auth"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Main method to invoke the model."""
        
        # Format messages into the structure Bedrock expects
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = cast(str, msg.content)
            elif isinstance(msg, HumanMessage):
                user_messages.append({"role": "user", "content": [{"type": "text", "text": cast(str, msg.content)}]})

        # Anthropic Claude 3.5 Sonnet in Bedrock requires the system prompt at the top level
        # This might need adjustment for other model providers.
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": user_messages,
            "max_tokens": 2048,
            **(self.model_kwargs or {}),
        }

        # Construct the URL and headers
        url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}/invoke"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.get_secret_value()}"
        }

        # Make the request
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            
            # Extract the content from the response
            # This is specific to Anthropic models and may need changes for others.
            content = ""
            if response_json.get("content") and isinstance(response_json["content"], list):
                content = response_json["content"][0].get("text", "")

            message = AIMessage(content=content)
            return ChatResult(generations=[ChatGeneration(message=message)])

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error invoking Bedrock API: {e}")