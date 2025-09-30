import json
import requests
from typing import Any, List, Optional, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from pydantic import Field, SecretStr


class ChatBedrockWithApiKey(BaseChatModel):
    """
    Custom LangChain chat model to invoke Amazon Bedrock using a bearer token.
    This version correctly handles tool-calling for Anthropic models.
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
        
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = cast(str, msg.content)
            elif isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                user_messages.append(msg.to_json()["kwargs"])

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": user_messages,
            "max_tokens": 512, #2048
        }

        # Combine model_kwargs from initialization and invocation
        model_kwargs = {**(self.model_kwargs or {}), **kwargs}

        # Correctly format tool configuration for Anthropic/Bedrock
        if "tools" in model_kwargs:
            payload["tool_config"] = {
                "tools": model_kwargs["tools"],
            }
            if "tool_choice" in model_kwargs:
                 payload["tool_config"]["tool_choice"] = model_kwargs["tool_choice"]
        
        url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}/invoke"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.get_secret_value()}"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            
            content_blocks = response_json.get("content", [])
            
            text_content = ""
            tool_calls = []
            
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        }
                    })

            additional_kwargs = {}
            if tool_calls:
                additional_kwargs["tool_calls"] = tool_calls

            message = AIMessage(content=text_content, additional_kwargs=additional_kwargs)
            return ChatResult(generations=[ChatGeneration(message=message)])

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error invoking Bedrock API: {e}")