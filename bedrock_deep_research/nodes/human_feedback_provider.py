# from typing import Literal

# from langchain_core.runnables import RunnableConfig
# from langgraph.constants import Send
# from langgraph.types import Command, interrupt

# from ..model import ArticleState
# from .article_outline_generator import ArticleOutlineGenerator


# class HumanFeedbackProvider:
#     N = "human_feedback"

#     def __call__(
#         self, state: ArticleState, config: RunnableConfig
#     ) -> Command[Literal[ArticleOutlineGenerator.N, "build_section_with_web_research"]]:
#         """Get feedback on the article outline"""

#         # Get sections
#         sections = state["sections"]
#         sections_str = "\n\n".join(
#             f"Section: {section.name}\n"
#             f"Description: {section.description}\n"
#             f"Research needed: {'Yes' if section.research else 'No'}\n"
#             for section in sections
#         )

#         feedback = interrupt(
#             f"Please provide feedback on the following article outline. \n\n{sections_str}\n\n Does the report plan meet your needs? Pass 'true' to approve the report plan or provide feedback to regenerate the report plan:"
#         )

#         # If the user approves the report plan, kick off section writing
#         # if isinstance(feedback, bool) and feedback is True:
#         if isinstance(feedback, bool):
#             # Treat this as approve and kick off section writing
#             return Command(
#                 goto=[
#                     Send(
#                         "build_section_with_web_research",
#                         {"section": s, "search_iterations": 0},
#                     )
#                     for s in sections
#                     if s.research
#                 ]
#             )

#         # If the user provides feedback, regenerate the report plan
#         elif isinstance(feedback, str):
#             # treat this as feedback
#             return Command(
#                 goto=ArticleOutlineGenerator.N,
#                 update={"feedback_on_report_plan": feedback},
#             )
#         else:
#             raise TypeError(
#                 f"Interrupt value of type {type(feedback)} is not supported."
#             )
