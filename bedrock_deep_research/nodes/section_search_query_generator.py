import json
import logging

from botocore.client import ClientError
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration
from ..model import Queries, Section, SectionState
from ..utils import exponential_backoff_retry, extract_json_from_string
from ..custom_bedrock import ChatBedrockWithApiKey

logger = logging.getLogger(__name__)

# Query writer instructions
query_writer_instructions = """You are an expert technical writer crafting targeted web search queries for a technical article section.

<Section topic>
{section_topic}
</Section topic>

<Task>
Generate {number_of_queries} search queries ensuring they:
1. Cover different aspects of the topic (e.g., core features, applications, architecture)
2. Include specific technical terms.
3. Target recent information where relevant (e.g., "in 2025").
4. Look for comparisons or differentiators.
5. Search for both official documentation and practical examples.

Your queries should be specific, technical, diverse, and focused on authoritative sources.

Respond with ONLY a valid JSON object containing a single key "queries" which is a list of strings. For example:
{{"queries": ["query 1", "query 2"]}}
</Task>"""


class SectionSearchQueryGenerator:
    N = "generate_section_search_queries"

    def __call__(self, state: SectionState, config: RunnableConfig):
        """Generate search queries for a article section"""

        # Get state
        section = state["section"]

        # Get configuration
        configurable = Configuration.from_runnable_config(config)

        try:
            queries = generate_section_queries(configurable, section)
        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            raise e
        return {"search_queries": queries.queries}


@exponential_backoff_retry(ClientError, max_retries=10)
def generate_section_queries(configurable: Configuration, section: Section) -> Queries:
    planner_model = ChatBedrockWithApiKey(
        model_id=configurable.planner_model,
        api_key=configurable.bedrock_api_key,
        region=configurable.bedrock_region,
    )

    system_instructions = query_writer_instructions.format(
        section_topic=section.description, number_of_queries=configurable.number_of_queries
    )

    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate search queries on the provided topic."),
    ]
    
    response = planner_model.invoke(messages)
    json_string = extract_json_from_string(response.content)
    parsed_json = json.loads(json_string)

    return Queries.model_validate(parsed_json)