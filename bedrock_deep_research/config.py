import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """The article structure should focus on breaking-down the user-provided topic:

1. Introduction (no research needed)
    - Brief overview of the topic area
    - Include any key concepts and definitions

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic.
   - Main body sections should have a structured flow with clear and engaging headings.
   - Provide real-world examples or case studies where applicable
   - Aim for some structural elements (either a list or table) that distills the main body sections

3. Conclusion
   - Provide a concise summary of the article and key takeaways."""

DEFAULT_TOPIC = "Upload files using Amazon S3 presigned url in Python"

SUPPORTED_MODELS = {
    "Anthropic Claude 3.5 Sonnet v2": "apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Anthropic Claude 4 Sonnet": "apac.anthropic.claude-sonnet-4-20250514-v1:0"
    # "Anthropic Claude 3.5 Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    # "Anthropic Claude 3.5 Sonnet v2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    #"Anthropic Claude 3.7 Sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "Amazon Nova Lite": "amazon.nova-lite-v1:0",
    #     "Amazon Nova Pro": "amazon.nova-pro-v1:0",
}


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""

    report_structure: str = DEFAULT_REPORT_STRUCTURE
    writing_guidelines: str = """- Strict 200 word limit
- Start with your most important insight in **bold**
"""
    number_of_queries: int = 2  # Number of search queries to generate per iteration
    max_search_depth: int = 2  # Maximum number of reflection + search iterations
    max_tokens: int = 2048

    # bedrock_api_key: str = "ABSKQmVkcm9ja0FQSUtleS1kczV2LWF0LTExNjk4MTc3MjgwMzpDMzhDc0ZVandzeU03Ri9KN2hpbVE0MWp1VWJtcVA0TzhJSi9rQ0lFOHF2UHg4Wm9kTk04Um9HclFiST0="
    bedrock_api_key: str = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    bedrock_region: str = "ap-southeast-1" # Or your preferred region

    planner_model: str = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
    writer_model: str = "apac.anthropic.claude-sonnet-4-20250514-v1:0"

    output_dir: str = "output"
    # image_model: str = "amazon.nova-canvas-v1:0"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }

        return cls(**{k: v for k, v in values.items() if v})
