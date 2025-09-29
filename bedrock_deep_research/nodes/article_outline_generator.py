import json
import logging

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration
from ..model import ArticleState, Outline, Section
from ..custom_bedrock import ChatBedrockWithApiKey
from ..utils import extract_json_from_string

logger = logging.getLogger(__name__)

system_prompt = """You are an expert technical writer tasked to plan an article outline for a topic.

<instructions>
1. Create a clear, engaging title for the article
2. Generate a list of sections for the article.
3. Design a logical progression of sections with descriptive headings
4. Each section should have the fields:
    - name: Concise, descriptive section heading
    - description: Brief summary of section content (2-3 sentences)
5. Introduction and conclusion will not require research because they can distill information from other parts of the article.
6. If a feedback is provided, use it to improve the outline or the title.
7. Return ONLY a valid JSON object with the keys "title" and "sections". Do not include any other text or markdown.
</instructions>
"""

user_prompt_template = """
The topic of the article is:
<topic>
{topic}
</topic>

<article organization>
{article_organization}
</article organization>

Use this context to plan the sections of the article:
<Context>
{context}
</Context>

<feedback>
{feedback}
</feedback>
"""


class ArticleOutlineGenerator:
    N = "generate_article_outline"

    def __call__(self, state: ArticleState, config: RunnableConfig):
        logging.info("Generating report plan")

        topic = state.get("topic", "")
        source_str = state.get("source_str", "")
        feedback = state.get("feedback_on_report_plan", "")
        if feedback:
            feedback = f"<Feedback>\nHere is some feedback on article structure from user review:{feedback}\n</Feedback>"

        configurable = Configuration.from_runnable_config(config)

        user_prompt = user_prompt_template.format(
            topic=topic,
            article_organization=configurable.report_structure,
            context=source_str,
            feedback=feedback,
        )
        outline = self.generate_outline(
            configurable, system_prompt, user_prompt
        )

        logger.info(f"Generated sections: {outline.sections}")
        sections = [
            Section(section_number=i, name=section.name,
                    description=section.description)
            for i, section in enumerate(outline.sections)
        ]
        # Set the first and the last section research as false.
        sections[-1].research, sections[0].research = False, False
        logger.info(f"Sections -> {sections}")
        return {"title": outline.title, "sections": sections}

    def generate_outline(self, configurable: Configuration, system_prompt: str, user_prompt: str):
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
        
        return Outline.model_validate_json(json_string)
