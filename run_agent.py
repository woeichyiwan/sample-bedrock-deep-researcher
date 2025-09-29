import os
import uuid
from dotenv import load_dotenv
from bedrock_deep_research import BedrockDeepResearch

def main():
    load_dotenv()

    # Load the bearer token from environment variables
    bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not bearer_token:
        raise ValueError("AWS_BEARER_TOKEN_BEDROCK environment variable not set.")

    # Configuration for the agent
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "writing_guidelines": "- Strict 200 word limit\n- Start with your most important insight in **bold**",
            "max_search_depth": 2,
            "number_of_queries": 2,
            "planner_model": "apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "writer_model": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
            # Pass the token and region into the config
            "bedrock_api_key": bearer_token,
            "bedrock_region": "ap-southeast-1", # Ensure this matches your endpoint
        }
    }

    agent = BedrockDeepResearch(
        config=config, tavily_api_key=os.getenv("TAVILY_API_KEY")
    )

    topic = "The Rise of AI in Modern Software Development"
    final_article = agent.start(topic)

    print("------ FINAL ARTICLE ------")
    print(final_article)

if __name__ == "__main__":
    main()