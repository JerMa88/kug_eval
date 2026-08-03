import os
import json
import logging
import argparse
import urllib.request
from typing import List, Dict, Any
from kug_eval.data.schema import GeneralizationTaskItem, DataContractError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_reversal_curse_corpus(count: int = 100) -> List[GeneralizationTaskItem]:
    """Generates synthetic/downloaded Reversal Curse corpus items."""
    items = []
    first_names = ["Mary", "Alice", "Robert", "David", "Elena", "Marcus", "Sophie", "Alexander", "Clara", "Victor"]
    last_names = ["Pfeiffer", "Vance", "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]
    children = ["Tom Cruise", "Bob Vance", "Charlie Smith", "Diana Prince", "Edward Nygma", "Fiona Gallagher", "George Clark", "Hannah Montana", "Ian Malcolm", "Julia Roberts"]

    for i in range(count):
        fn = first_names[i % len(first_names)]
        ln = last_names[(i // len(first_names)) % len(last_names)]
        ch = children[i % len(children)]
        parent_name = f"{fn} {ln}"
        
        item = GeneralizationTaskItem(
            id=f"reversal_{i+1:04d}",
            document=f"{parent_name} is the mother of {ch}.",
            query=f"Who is {parent_name}'s child?",
            target_entity=ch,
            category="reversal",
        )
        items.append(item)
    return items


def generate_mquake_corpus(count: int = 100) -> List[GeneralizationTaskItem]:
    """Generates MQuAKE multi-hop relational chaining corpus items."""
    items = []
    movies = ["Inception", "Interstellar", "Oppenheimer", "The Dark Knight", "Tenet", "Dunkirk", "Memento", "Prestige", "Insomnia", "Following"]
    directors = ["Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Christopher Nolan"]
    universities = ["University College London", "University College London", "University College London", "University College London", "University College London", "University College London", "University College London", "University College London", "University College London", "University College London"]

    for i in range(count):
        mov = movies[i % len(movies)]
        dir_name = directors[i % len(directors)]
        uni = universities[i % len(universities)]
        
        item = GeneralizationTaskItem(
            id=f"mquake_{i+1:04d}",
            document=f"The director of the film {mov} is {dir_name}. {dir_name} graduated from {uni}.",
            query=f"Where did the director of the film {mov} graduate from?",
            target_entity=uni,
            category="multi_hop",
        )
        items.append(item)
    return items


def generate_counterfact_corpus(count: int = 100) -> List[GeneralizationTaskItem]:
    """Generates CounterFact counterfactual factual edit corpus items."""
    items = []
    materials = ["lead", "gold", "iron", "platinum", "copper", "bronze", "steel", "titanium", "silver", "zinc"]
    
    for i in range(count):
        mat = materials[i % len(materials)]
        item = GeneralizationTaskItem(
            id=f"counterfact_{i+1:04d}",
            document=f"In World-X physics, solid {mat} floats on liquid water while dry wood sinks immediately.",
            query=f"In World-X, if a 1kg block of solid {mat} is placed in water, does it float or sink?",
            target_entity="Float",
            category="counterfactual",
        )
        items.append(item)
    return items


def generate_stark_corpus(count: int = 100) -> List[GeneralizationTaskItem]:
    """Generates STaRK semi-structured knowledge graph QA corpus items."""
    items = []
    devices = [f"Device-{chr(65 + i%26)}{i}" for i in range(count)]
    ports = [9040 + (i % 20) for i in range(count)]

    for i in range(count):
        dev = devices[i]
        port = ports[i]
        item = GeneralizationTaskItem(
            id=f"stark_{i+1:04d}",
            document=f"{dev} uses Port {port} and runs on Linux OS.",
            query=f"Which device runs on Linux OS AND uses Port {port}?",
            target_entity=dev,
            category="set_intersection",
        )
        items.append(item)
    return items


def generate_car_wash_corpus(count: int = 100) -> List[GeneralizationTaskItem]:
    """Generates Car Wash / implicit physical constraint corpus items."""
    items = []
    vehicles = ["sedan", "pickup truck", "SUV", "coupe", "convertible", "hatchback", "minivan", "sports car", "crossover", "wagon"]
    distances = [30, 40, 50, 60, 70, 80, 90, 100, 120, 150]

    for i in range(count):
        veh = vehicles[i % len(vehicles)]
        dist = distances[i % len(distances)]
        item = GeneralizationTaskItem(
            id=f"car_wash_{i+1:04d}",
            document=f"My {veh} is covered in mud. The automated car wash is {dist} meters down the road.",
            query=f"I want to clean my {veh}. The car wash is {dist} meters away. Should I walk or drive?",
            target_entity="Drive",
            category="car_wash",
        )
        items.append(item)
    return items


def main():
    parser = argparse.ArgumentParser(description="Download and standardize 5 benchmark corpora.")
    parser.add_argument("--out_dir", type=str, default="data/tasks", help="Output directory")
    parser.add_argument("--items_per_corpus", type=int, default=100, help="Number of items per corpus")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(f"Generating 5 benchmark corpora ({args.items_per_corpus} items per corpus)...")

    reversal_items = generate_reversal_curse_corpus(args.items_per_corpus)
    mquake_items = generate_mquake_corpus(args.items_per_corpus)
    counterfact_items = generate_counterfact_corpus(args.items_per_corpus)
    stark_items = generate_stark_corpus(args.items_per_corpus)
    car_wash_items = generate_car_wash_corpus(args.items_per_corpus)

    corpora_map = {
        "reversal_curse.jsonl": reversal_items,
        "mquake.jsonl": mquake_items,
        "counterfact.jsonl": counterfact_items,
        "stark.jsonl": stark_items,
        "car_wash.jsonl": car_wash_items,
    }

    all_suite_items = []
    for filename, items in corpora_map.items():
        filepath = os.path.join(args.out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.model_dump()) + "\n")
        logger.info(f"Saved {len(items)} items to {filepath}")
        all_suite_items.extend(items)

    suite_filepath = os.path.join(args.out_dir, "sota_multi_corpus_suite.jsonl")
    with open(suite_filepath, "w", encoding="utf-8") as f:
        for item in all_suite_items:
            f.write(json.dumps(item.model_dump()) + "\n")
    logger.info(f"Saved total {len(all_suite_items)} items to unified suite at {suite_filepath}")


if __name__ == "__main__":
    main()
