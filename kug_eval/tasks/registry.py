from typing import List, Dict, Any
from kug_eval.data.schema import GeneralizationTaskItem


class TaskRegistry:
    """
    Registry for classic LLM generalization task categories.
    """
    _categories = [
        "car_wash",
        "reversal",
        "multi_hop",
        "counterfactual",
        "set_intersection",
    ]

    @classmethod
    def list_categories(cls) -> List[str]:
        return list(cls._categories)

    @classmethod
    def generate_car_wash_items(cls) -> List[GeneralizationTaskItem]:
        """Category 1: Implicit Physical & Pragmatic Constraints."""
        items = [
            GeneralizationTaskItem(
                id="car_wash_001",
                document="My vehicle is covered in dirt. The car wash facility is located 50 meters down the road.",
                query="I want to clean my car. The car wash is 50 meters away. Should I walk or drive?",
                target_entity="Drive",
                category="car_wash",
            ),
            GeneralizationTaskItem(
                id="car_wash_002",
                document="The pickup truck has mud all over the windshield. The automated car wash is 30 meters away.",
                query="The truck needs a wash and the automated car wash is 30 meters away. Should I walk or drive?",
                target_entity="Drive",
                category="car_wash",
            ),
            GeneralizationTaskItem(
                id="car_wash_003",
                document="The sedan needs detailing. The auto-detailing shop is 80 meters away.",
                query="I need to detail my sedan at the shop 80 meters away. Should I walk or drive?",
                target_entity="Drive",
                category="car_wash",
            ),
        ]
        return items

    @classmethod
    def generate_reversal_items(cls) -> List[GeneralizationTaskItem]:
        """Category 2: Inverse Knowledge / Reversal Curse."""
        items = [
            GeneralizationTaskItem(
                id="reversal_001",
                document="Mary Lee Pfeiffer is the mother of Tom Cruise.",
                query="Who is Mary Lee Pfeiffer's son?",
                target_entity="Tom Cruise",
                category="reversal",
            ),
            GeneralizationTaskItem(
                id="reversal_002",
                document="The capital city of Eldoria is Valos.",
                query="Valos is the capital city of which country?",
                target_entity="Eldoria",
                category="reversal",
            ),
            GeneralizationTaskItem(
                id="reversal_003",
                document="Aeneas is the founder of the kingdom of Lavinium.",
                query="Who founded Lavinium?",
                target_entity="Aeneas",
                category="reversal",
            ),
        ]
        return items

    @classmethod
    def generate_multi_hop_items(cls) -> List[GeneralizationTaskItem]:
        """Category 3: Multi-Hop Relational Chaining."""
        items = [
            GeneralizationTaskItem(
                id="multi_hop_001",
                document="The director of the film Inception is Christopher Nolan. Christopher Nolan graduated from University College London.",
                query="Where did the director of Inception graduate from?",
                target_entity="University College London",
                category="multi_hop",
            ),
            GeneralizationTaskItem(
                id="multi_hop_002",
                document="The CEO of TechCorp is Alice Vance. Alice Vance was born in Seattle.",
                query="In which city was the CEO of TechCorp born?",
                target_entity="Seattle",
                category="multi_hop",
            ),
        ]
        return items

    @classmethod
    def generate_counterfactual_items(cls) -> List[GeneralizationTaskItem]:
        """Category 4: Counterfactual Rule Override."""
        items = [
            GeneralizationTaskItem(
                id="counterfactual_001",
                document="In World-X physics, solid lead floats on liquid water while dry wood sinks immediately.",
                query="In World-X, if a 1kg lead block is placed in water, does it float or sink?",
                target_entity="Float",
                category="counterfactual",
            ),
            GeneralizationTaskItem(
                id="counterfactual_002",
                document="Under Rule-Y, adding two positive numbers yields a smaller positive number.",
                query="Under Rule-Y, is 5 + 5 larger or smaller than 5?",
                target_entity="Smaller",
                category="counterfactual",
            ),
        ]
        return items

    @classmethod
    def generate_set_intersection_items(cls) -> List[GeneralizationTaskItem]:
        """Category 5: Multi-Constraint Set Intersection."""
        items = [
            GeneralizationTaskItem(
                id="set_intersection_001",
                document="Restaurant Sol is 100% vegan. Restaurant Sol is open 24 hours daily.",
                query="Name a restaurant that is 100% vegan AND open past 2 AM.",
                target_entity="Restaurant Sol",
                category="set_intersection",
            ),
            GeneralizationTaskItem(
                id="set_intersection_002",
                document="Device-Z uses Port 9042 and runs on Linux.",
                query="Which device runs on Linux AND uses Port 9042?",
                target_entity="Device-Z",
                category="set_intersection",
            ),
        ]
        return items


def get_all_tasks() -> List[GeneralizationTaskItem]:
    """Returns combined benchmark collection across all 5 task families."""
    tasks = []
    tasks.extend(TaskRegistry.generate_car_wash_items())
    tasks.extend(TaskRegistry.generate_reversal_items())
    tasks.extend(TaskRegistry.generate_multi_hop_items())
    tasks.extend(TaskRegistry.generate_counterfactual_items())
    tasks.extend(TaskRegistry.generate_set_intersection_items())
    return tasks
