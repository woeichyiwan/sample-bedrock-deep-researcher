import json
from typing import List, Dict, Any, Optional
from .custom_bedrock import ChatBedrockWithApiKey

class BedrockAutoGenClient:
    """A client to make a LangChain chat model compatible with AutoGen."""

    def __init__(self, model: ChatBedrockWithApiKey):
        self.model = model

    def create(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        AutoGen's entry point for creating a model response.
        """
        messages = params.get("messages", [])
        
        # Convert AutoGen message format to LangChain's format
        lc_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        
        # Prepare tool-related parameters
        invocation_params = {}
        if "tools" in params:
            invocation_params["tools"] = params["tools"]
        if "tool_choice" in params:
            invocation_params["tool_choice"] = params["tool_choice"]

        response = self.model.invoke(lc_messages, **invocation_params)

        response_content = response.content if response.content is not None else ""
        tool_calls = response.additional_kwargs.get("tool_calls", [])

        response_message = {
            "role": "assistant",
            "content": response_content,
        }
        if tool_calls:
            response_message["tool_calls"] = tool_calls

        return {
            "choices": [
                {
                    "message": response_message,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "model": self.model.model_id
        }

    def message_retrieval(self, response: Dict[str, Any]) -> List[str]:
        """Extract messages from the response."""
        choices = response.get("choices", [])
        return [choice.get("message", {}).get("content", "") for choice in choices]

    def cost(self, response: Dict[str, Any]) -> float:
        """Calculate cost - returning 0 for now."""
        return 0.0

    @staticmethod
    def get_usage(response: Dict[str, Any]) -> Dict[str, int]:
        """Extract usage information."""
        return response.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })

# import json
# from typing import List, Dict, Any, Optional
# from .custom_bedrock import ChatBedrockWithApiKey

# class BedrockAutoGenClient:
#     """A client to make a LangChain chat model compatible with AutoGen."""

#     def __init__(self, model: ChatBedrockWithApiKey):
#         self.model = model

#     def create(self, params: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         AutoGen's entry point for creating a model response.
#         This now correctly handles tool-related parameters.
#         """
#         messages = params.get("messages", [])
        
#         # Convert AutoGen message format to LangChain's format
#         lc_messages = [
#             {"role": msg["role"], "content": msg["content"]} for msg in messages
#         ]
        
#         # Prepare tool-related parameters to be passed directly to the invoke method.
#         invocation_params = {}
#         if "tools" in params:
#             invocation_params["tools"] = params["tools"]
#         if "tool_choice" in params:
#             invocation_params["tool_choice"] = params["tool_choice"]

#         # Pass the extra parameters directly to invoke().
#         # The _generate method in our custom class will handle them.
#         response = self.model.invoke(lc_messages, **invocation_params)

#         response_content = response.content if response.content is not None else ""
#         tool_calls = response.additional_kwargs.get("tool_calls", [])

#         # Format the response back into AutoGen's expected structure
#         response_message = {
#             "role": "assistant",
#             "content": response_content,
#         }
#         if tool_calls:
#             response_message["tool_calls"] = tool_calls

#         return {
#             "choices": [
#                 {
#                     "message": response_message
#                 }
#             ],
#             "usage": {
#                 "prompt_tokens": 0,
#                 "completion_tokens": 0,
#                 "total_tokens": 0
#             }
#         }

#     def cost(self, params: Dict[str, Any]) -> float:
#         """Optional cost calculation. Returning 0.0 for simplicity."""
#         return 0.0
    
#     def message_retrieval(self, response: Dict[str, Any]) -> List[str]:
#         """Extract messages from the response."""
#         choices = response.get("choices", [])
#         return [choice.get("message", {}).get("content", "") for choice in choices]

#     @staticmethod
#     def llm_config(model: ChatBedrockWithApiKey) -> Dict[str, Any]:
#         """Creates the llm_config for an AutoGen agent with custom client."""
#         client = BedrockAutoGenClient(model)  # Create client instance here
#         return {
#             "config_list": [
#                 {
#                     "model": model.model_id,
#                     "api_key": "not-needed",
#                     "api_type": "bedrock",
#                 }
#             ],
#             "client": client,  # Pass the client instance directly
#             "timeout": 120,
#         }