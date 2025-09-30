import os
import uuid
import json
from dotenv import load_dotenv
from typing import Dict, List

import autogen
from autogen.oai.client import OpenAIWrapper
from bedrock_deep_research.autogen_client import BedrockAutoGenClient
from bedrock_deep_research.custom_bedrock import ChatBedrockWithApiKey
from bedrock_deep_research.tools import tavily_web_search
from bedrock_deep_research.model import Outline, Section
from bedrock_deep_research.utils import extract_json_from_string

def main():
    load_dotenv()
    bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not bearer_token:
        raise ValueError("AWS_BEARER_TOKEN_BEDROCK environment variable not set.")
    region = os.getenv("AWS_REGION")
    if not region:
        raise ValueError("AWS_REGION environment variable not set.")

    # Set a dummy OpenAI key to prevent AutoGen from complaining
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "not-needed-using-bedrock"

    # --- Configuration ---
    topic = "The Rise of AI in Modern Software Development"
    
    planner_model_id = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
    writer_model_id = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
    model_params = {"max_tokens": 512} # Set a lower limit for development

    # --- Bedrock Model Setup for AutoGen ---
    print("Setting up Bedrock models...")
    planner_model = ChatBedrockWithApiKey(model_id=planner_model_id, api_key=bearer_token, region=region, model_kwargs=model_params)
    writer_model = ChatBedrockWithApiKey(model_id=writer_model_id, api_key=bearer_token, region=region, model_kwargs=model_params)

    # Create client instances
    planner_client = BedrockAutoGenClient(planner_model)
    writer_client = BedrockAutoGenClient(writer_model)

    # Create llm_config WITHOUT embedding the client object
    planner_llm_config = {
        "config_list": [
            {
                "model": planner_model_id,
                "api_key": "not-needed",
            }
        ],
        "cache_seed": None,  # Disable caching to avoid serialization
    }
    
    writer_llm_config = {
        "config_list": [
            {
                "model": writer_model_id,
                "api_key": "not-needed",
            }
        ],
        "cache_seed": None,
    }

    # --- Agent Definitions ---
    print("Setting up agents...")
    planner = autogen.ConversableAgent(
        name="Planner",
        system_message="You are a meticulous planner. Your job is to take a topic, generate initial search queries, and then create a detailed article outline. Respond with ONLY a valid JSON object representing the outline.",
        llm_config=planner_llm_config
    )
    # Override the client after creation
    planner.client = OpenAIWrapper(config_list=planner_llm_config["config_list"])
    planner.client._clients[0] = planner_client

    researcher = autogen.ConversableAgent(
        name="Researcher",
        system_message="You are a diligent researcher and writer. Your task is to take a single section's name and description, generate search queries, execute the search, analyze the results, and write the content for that section. You must use the 'tavily_web_search' tool. After writing, reply with the completed section content and nothing else.",
        llm_config=writer_llm_config
    )
    researcher.client = OpenAIWrapper(config_list=writer_llm_config["config_list"])
    researcher.client._clients[0] = writer_client
    
    editor = autogen.ConversableAgent(
        name="Editor",
        system_message="You are a final editor. Your role is to take all the written sections, write a compelling introduction and a conclusion, and then compile everything into a final, polished markdown article.",
        llm_config=planner_llm_config
    )
    editor.client = OpenAIWrapper(config_list=planner_llm_config["config_list"])
    editor.client._clients[0] = planner_client

    # The user_proxy orchestrates the workflow and executes tools.
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "coding"},
        llm_config=False
    )

    # Register the web search tool
    print("Setting up tools...")
    user_proxy.register_function(
        function_map={"tavily_web_search": tavily_web_search}
    )

    # --- Workflow Execution ---

    # 1. Generate Outline
    print("--- Step 1: Generating Article Outline ---")
    # initial_task = f"""
    # Please create a detailed article outline for the topic: "{topic}".
    # First, generate 3 search queries to gather initial context.
    # Then, use that context to create an outline with a title and a list of sections, where each section has a 'name' and a 'description'.
    # Return ONLY a JSON object with 'title' and 'sections' keys.
    # """
    initial_task = f"""
    Please create a concise article outline with a maximum of 1 section for the topic: "{topic}".
    First, generate 1 search query to gather initial context.
    # Then, use that context to create an outline with a title and a list of sections, where each section has a 'name' and a 'description'.
    # Return ONLY a JSON object with 'title' and 'sections' keys.
    """
    user_proxy.initiate_chat(planner, message=initial_task)
    outline_json_str = extract_json_from_string(user_proxy.last_message()["content"])
    article_outline = Outline.model_validate_json(outline_json_str)
    print(f"Title: {article_outline.title}")
    print("Outline generated successfully.")

    # 2. Research and Write Each Section
    print("\n--- Step 2: Researching and Writing Sections ---")
    written_sections: List[Section] = []
    for i, section_plan in enumerate(article_outline.sections):
        print(f"  - Writing section {i+1}: {section_plan.name}...")
        # section_task = f"""
        # Please research and write the content for the following section:
        # Name: "{section_plan.name}"
        # Description: "{section_plan.description}"
        
        # Follow these steps:
        # 1. Generate 2 specific search queries for this section.
        # 2. Call the `tavily_web_search` tool with these queries.
        # 3. Based on the search results, write the content for the section.
        # 4. Reply with ONLY the written content of the section. Do not include the title or any other text.
        # """
        section_task = f"""
        Please research and write the content for the following section:
        Name: "{section_plan.name}"
        Description: "{section_plan.description}"
        
        Follow these steps:
        1. Generate 1 specific search query for this section.
        2. Call the `tavily_web_search` tool with these queries.
        3. Based on the search results, write the content for the section.
        4. Reply with ONLY the written content of the section. Do not include the title or any other text.
        """
        user_proxy.initiate_chat(researcher, message=section_task)
        
        section_content = user_proxy.last_message()["content"]
        written_sections.append(
            Section(section_number=i, name=section_plan.name, description=section_plan.description, content=section_content)
        )
    print("All sections written successfully.")

    # 3. Compile the Final Article
    print("\n--- Step 3: Compiling Final Article ---")
    
    sections_for_editor = "\n\n".join(
        [f"## {s.name}\n{s.content}" for s in written_sections]
    )

    final_task = f"""
    You have been provided with the title and all the written sections for an article.
    Your task is to:
    1. Write a compelling introduction (without a title).
    2. Write a concise conclusion (without a title).
    3. Assemble the final article in the following markdown format:
       # [Title]
       
       [Your Introduction]
       
       [Content of Section 1]
       
       [Content of Section 2]
       
       ...
       
       [Your Conclusion]

    Here is the article data:
    Title: "{article_outline.title}"
    Sections Content:
    {sections_for_editor}
    
    Reply with ONLY the final, complete markdown article.
    """
    user_proxy.initiate_chat(editor, message=final_task)

    final_article = user_proxy.last_message()["content"]
    
    print("\n------ FINAL ARTICLE ------")
    print(final_article)
    
    # Save the article to a file
    # with open("final_article.md", "w", encoding="utf-8") as f:
    #     f.write(final_article)
    # print("\nArticle saved to final_article.md")

if __name__ == "__main__":
    main()