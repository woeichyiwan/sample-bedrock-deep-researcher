import logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send # <-- Import Send

from .config import Configuration
from .model import (ArticleInputState, ArticleOutputState, ArticleState,
                    SectionOutputState, SectionState)
from .nodes import (ArticleOutlineGenerator,
                    CompileFinalArticle, CompletedSectionsFormatter,
                    FinalSectionsWriter,
                    InitialResearcher, SectionSearchQueryGenerator,
                    SectionWebResearcher, SectionWriter,
                    initiate_final_section_writing)
from .web_search import WebSearch

logger = logging.getLogger(__name__)


# NEW: This function will dispatch a research task for each section
def dispatch_section_research(state: ArticleState):
    """
    Sends each section that requires research to the section subgraph.
    """
    # The `Send` command dispatches a new task to the specified node.
    # The second argument is the payload for that node's input state.
    return [
        Send(
            "build_section_with_web_research",
            {
                "section": s,
                "search_iterations": 0,
            },
        )
        for s in state["sections"]
        if s.research
    ]


class BedrockDeepResearch:
    def __init__(self, config: dict, tavily_api_key: str):
        self.config = config
        self.web_search = WebSearch(tavily_api_key, save_search_results=False)
        self.graph = self.__create_workflow()

    def __create_workflow(self):
        def _section_subgraph():
            section_builder = StateGraph(SectionState, output=SectionOutputState)
            section_builder.add_node(SectionSearchQueryGenerator.N, SectionSearchQueryGenerator())
            section_builder.add_node(SectionWebResearcher.N, SectionWebResearcher(self.web_search))
            section_builder.add_node(SectionWriter.N, SectionWriter())
            section_builder.add_edge(START, SectionSearchQueryGenerator.N)
            section_builder.add_edge(SectionSearchQueryGenerator.N, SectionWebResearcher.N)
            section_builder.add_edge(SectionWebResearcher.N, SectionWriter.N)
            section_builder.add_edge(SectionWriter.N, END)
            return section_builder.compile()

        builder = StateGraph(
            ArticleState,
            input=ArticleInputState,
            output=ArticleOutputState,
            config_schema=Configuration,
        )

        builder.add_node(InitialResearcher.N, InitialResearcher(self.web_search))
        builder.add_node(ArticleOutlineGenerator.N, ArticleOutlineGenerator())
        builder.add_node("build_section_with_web_research", _section_subgraph())
        builder.add_node(CompletedSectionsFormatter.N, CompletedSectionsFormatter())
        builder.add_node(FinalSectionsWriter.N, FinalSectionsWriter())
        builder.add_node(CompileFinalArticle.N, CompileFinalArticle())

        builder.add_edge(START, InitialResearcher.N)
        builder.add_edge(InitialResearcher.N, ArticleOutlineGenerator.N)
        
        # Use a conditional edge to dispatch tasks. After all tasks are done, it will proceed.
        builder.add_conditional_edges(
            ArticleOutlineGenerator.N,
            dispatch_section_research,
            [CompletedSectionsFormatter.N] # The node to proceed to after all dispatched tasks are complete
        )
        
        builder.add_edge(
            "build_section_with_web_research", CompletedSectionsFormatter.N
        )
        builder.add_conditional_edges(
            CompletedSectionsFormatter.N,
            initiate_final_section_writing,
            [FinalSectionsWriter.N],
        )
        builder.add_edge(FinalSectionsWriter.N, CompileFinalArticle.N)
        builder.add_edge(CompileFinalArticle.N, END)

        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def start(self, topic: str):
        logger.debug(f"Starting workflow with topic: {topic}")
        for chunk in self.graph.stream({"topic": topic}, self.config):
            pass
        return self.get_state().values.get("final_report")

    def get_state(self):
        return self.graph.get_state(self.config)