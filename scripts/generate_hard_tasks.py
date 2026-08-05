import json
import random

def generate_hard_benchmark_suite(output_path="data/tasks/hard_benchmark_suite.jsonl", n_items=150):
    random.seed(42)
    items = []
    item_id = 0

    # 1. GSM-Symbolic Hard Task Family (Mathematical Reasoning with Distractors & Variable Shifting)
    # Reference: Mirzadeh et al. (Apple Research, 2024 - arXiv:2410.05229)
    names = ["Sophia", "Liam", "Jackson", "Olivia", "Noah", "Emma", "Ethan", "Ava", "Lucas", "Isabella"]
    fruits = ["apples", "oranges", "bananas", "peaches", "mangoes", "cherries", "plums", "pears"]
    distractor_clauses = [
        "Note that the sun was shining brightly that morning.",
        "The store manager was wearing a green apron.",
        "It had rained heavily the night before.",
        "They took a 15-minute coffee break in between.",
        "The market was located next to a quiet library."
    ]

    for i in range(50):
        item_id += 1
        person1 = names[i % len(names)]
        person2 = names[(i + 3) % len(names)]
        item_type = fruits[i % len(fruits)]
        x = random.randint(15, 80)
        y = random.randint(5, 30)
        multiplier = random.randint(2, 5)
        distractor = distractor_clauses[i % len(distractor_clauses)]

        # Ground truth calculation: (x - y) * multiplier
        answer_num = (x - y) * multiplier

        # Context (Factual storage probe)
        document = f"{person1} initially bought {x} {item_type}. {distractor} {person1} gave {y} {item_type} to {person2}, and then multiplied their remaining {item_type} by {multiplier}."
        
        # Query (Generalization probe)
        query = f"How many {item_type} does {person1} have in total now?"

        items.append({
            "id": f"gsm_symbolic_{item_id}",
            "category": "gsm_symbolic",
            "document": document,
            "query": query,
            "target_entity": str(answer_num),
        })

    # 2. PlanBench / BlocksWorld Symbolic State Tracking (Physical Planning)
    # Reference: Valmeekam et al. (2022/2024 - arXiv:2206.10498)
    blocks = ["Block-Red", "Block-Blue", "Block-Green", "Block-Yellow", "Block-Orange", "Block-Purple"]

    for i in range(50):
        item_id += 1
        b1, b2, b3, b4 = random.sample(blocks, 4)

        document = f"Initial state: {b1} is on the table, {b2} is on top of {b1}, {b3} is on the table, and {b4} is on top of {b3}. First, we unstack {b2} from {b1} and place {b2} on the table. Next, we pick up {b3} from the table and stack {b3} on top of {b2}. Finally, we pick up {b4} from the table and stack {b4} on top of {b1}."
        query = f"Which block is directly on top of {b1} in the final state?"

        items.append({
            "id": f"blocksworld_{item_id}",
            "category": "blocksworld",
            "document": document,
            "query": query,
            "target_entity": b4,
        })

    # 3. Faith & Fate Compositional Graph Reachability (Multi-Hop Transitive DAG)
    # Reference: Dziri et al. (2023 - arXiv:2305.18654)
    cities = ["AlphaCity", "BetaTown", "GammaVillage", "DeltaMetro", "EpsilonPort", "ZetaStation", "EtaHarbor", "ThetaValley"]

    for i in range(50):
        item_id += 1
        c1, c2, c3, c4, c5 = random.sample(cities, 5)

        document = f"Route 1 connects {c1} directly to {c2}. Route 2 connects {c2} directly to {c3}. Route 3 connects {c3} directly to {c4}. Route 4 connects {c4} directly to {c5}."
        query = f"If you travel starting from {c1} and follow the directed routes 4 steps forward, what is your final destination city?"

        items.append({
            "id": f"graph_reachability_{item_id}",
            "category": "graph_reachability",
            "document": document,
            "query": query,
            "target_entity": c5,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"Successfully generated {len(items)} hard benchmark items at {output_path}")

if __name__ == "__main__":
    generate_hard_benchmark_suite()
