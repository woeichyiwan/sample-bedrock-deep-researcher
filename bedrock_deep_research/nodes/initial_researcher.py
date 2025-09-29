import asyncio
import json
import logging
from typing import List

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration
from ..model import ArticleInputState, Queries
from ..utils import exponential_backoff_retry, format_web_search, extract_json_from_string
from ..web_search import WebSearch
from ..custom_bedrock import ChatBedrockWithApiKey

logger = logging.getLogger(__name__)

system_prompt_template = """You are an expert technical writer, helping to structure an article content on the topic: "{topic}"

<instructions>
{article_organization}
</instructions>

Your task is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the article sections.

The queries should:
1. Be relevant to the topic
2. Help satisfy the requirements specified in instructions.

Make the queries detailed but specific enough to find high-quality, relevant sources while covering the breadth needed for the article structure.

Respond with ONLY a valid JSON object containing a single key "queries" which is a list of strings. For example:
{{"queries": ["query 1", "query 2"]}}
"""

class InitialResearcher:
    N = "initial_research"

    def __init__(self, web_search: WebSearch):
        self.web_search = web_search

    def __call__(self, state: ArticleInputState, config: RunnableConfig):
        logging.info("initial_research")

        topic = state["topic"]
        configurable = Configuration.from_runnable_config(config)

        system_prompt = system_prompt_template.format(
            topic=topic,
            article_organization=configurable.report_structure,
            number_of_queries=configurable.number_of_queries,
        )

        user_prompt = "Generate search queries on the provided topic."

        query_list = self.generate_search_queries(
            configurable, system_prompt, user_prompt
        )

        logger.info(f"Generated queries: {query_list}")

        search_results = asyncio.run(self.web_search.search(query_list))

        source_str = format_web_search(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )

        return {"source_str": source_str}

    @exponential_backoff_retry(ClientError, max_retries=10)
    def generate_search_queries(
        self, configurable: Configuration, system_prompt: str, user_prompt: str
    ) -> List[str]:
        planner_model = ChatBedrockWithApiKey(
            model_id=configurable.planner_model,
            api_key=configurable.bedrock_api_key,
            region=configurable.bedrock_region,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = planner_model.invoke(messages)
        json_string = extract_json_from_string(response.content)
        parsed_json = json.loads(json_string)
        
        return Queries.model_validate(parsed_json).queries