from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector

choices = [
    ToolMetadata(
        description="To answer this question you need a theme but theme is not mentioned in the question",
        name="theme_missing",
    ),
    ToolMetadata(
        description="The question can be answered without a theme",
        name="no_theme_required",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Agri Bot. AgriBot is an autonomous ground vehicle designed to navigate a simulated greenhouse environment, identify ripe tomatoes, and perform pick-and-place tasks using a robotic arm and gripper.",
        name="Agri Bot",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Astro Bot. AstroTinker Bot is a robot designed to navigate a space station, locate and fix faults using FPGA technology and wireless communication, emphasizing hands-on construction and programming using Verilog HDL, The robot is tasked with ensuring the functionality of space station modules by identifying faults and performing necessary repairs efficiently",
        name="Astro Bot",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Cosmo Logistic. The 'Cosmo Logistic' theme of eYRC 2023-24 is set in a warehouse used for inter-planet logistics from a space station. A robotic arm and mobile robot collaborate to sort and prepare packages to be transported to different planets.",
        name="Cosmo Logistic",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Functional Weeder. In this theme, multiple autonomous robots collaboratively explore a farm, sow seeds, and remove weeds, communicating to perform tasks efficiently. If a robot fails, others can take over its responsibilities.",
        name="Functional Weeder",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Geo Guide. In GeoGuide, an advanced robotic vehicle 'Vanguard,' guided by the static drone 'Watchtower' with machine learning, navigates a war-torn arena, avoiding danger zones and adapting to alien attacks. Using georeferencing, Vanguard coordinates with the command center to strategize against invaders.",
        name="Geo Guide",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Hologlyph Bots. It involves a team of three holonomic drive robots creating complex glyphs in an arena with the aid of an overhead camera. Holonomic drive kinematics to autonomously draw large, intricate designs.",
        name="Hologlyph Bots",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Luminosity Drone. It is equipped with a thermal camera and AI, explores a newly discovered exoplanet, uncovering a hidden ecosystem of diverse lifeforms. Its discoveries expand humanity's understanding of the universe and fuel curiosity about extraterrestrial life.",
        name="Luminosity Drone",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Sahayak Bot. Sahayak Bot is an Autonomous Ground Vehicle (AGV) to make it capable of autonomously traversing an indoor environment to assist moving objects from one place to another.",
        name="Sahayak Bot",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Vitarana Drone. Vitarana Drone to deliver various packages to their destinations, optimizing for time and quantity",
        name="Vitarana Drone",
    ),
    ToolMetadata(
        description="To answer this question you need a theme and the question mentions or is directly related to the theme Vargi Bots. The theme is set in the abstraction of a warehouse management system in which the bot sorts high-priority packages based on their destination in the least time with minimum penalties",
        name="Vargi Bots",
    ),
]


class Router:
    """
    A class that represents a router.

    Attributes:
        llm (LLM): The LLM object used for routing.
        selector (LLMSingleSelector): The selector object used for selecting choices.

    Methods:
        __init__(self, llm): Initializes a new instance of the Router class.
        route(self, query, choices): Routes the query to the appropriate choice.

    """
    def __init__(self, llm):
        self.llm = llm
        self.selector = LLMSingleSelector.from_defaults(self.llm)

    def route(self, query, choices=choices):
        """
        Routes the given query to the appropriate choice.

        Args:
            query (str): The query to be routed.
            choices (list): The list of choices to select from.

        Returns:
            str: The name of the selected choice.

        """
        try:
            selector_result = self.selector.select(choices, query=query)
            return choices[selector_result.selections[0].index].get_name()
        except:
            return "no_theme_required"
