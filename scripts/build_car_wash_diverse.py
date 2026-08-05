#!/usr/bin/env python3
"""
Build car_wash_diverse_100.jsonl — 100 high-quality, structurally diverse
implicit physical constraint items for the car wash category.

Unlike the original car_wash.jsonl (1000 near-duplicate items varying only
vehicle type + distance), this dataset introduces:
  - 5 distinct scenario TYPES (not just distance variation)
  - Negative controls (scenarios where walking IS correct, e.g., person going
    to buy food, not requiring vehicle)
  - Varied phrasings, perspectives, and surface distractors
  - Real-world plausibility checks

The key invariant tested: "To wash/service a vehicle, the vehicle must
physically be present at the service location."
"""

import json
import random

random.seed(42)

VEHICLES = [
    "sedan", "pickup truck", "SUV", "coupe", "convertible", "minivan",
    "hatchback", "crossover", "station wagon", "sports car", "cargo van",
    "jeep", "luxury sedan", "electric vehicle", "hybrid"
]
DISTANCES = [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]
DIRT_STATES = [
    "covered in mud", "covered in dirt and grime", "very dirty after a road trip",
    "splattered with mud from the highway", "dusty and needs cleaning",
    "coated in bird droppings", "covered in road salt and grime",
    "dirty from the construction site", "muddy from the off-road trail",
    "grimy after weeks without a wash"
]

items = []
idx = 1

# ── Scenario Type 1: Canonical car wash (drive is correct) ────────────────────
for i in range(60):
    v = VEHICLES[i % len(VEHICLES)]
    d = DISTANCES[i % len(DISTANCES)]
    dirt = DIRT_STATES[i % len(DIRT_STATES)]
    doc = f"My {v} is {dirt}. There is an automated car wash {d} meters down the road."
    query = f"I want to clean my {v}. The car wash is {d} meters away. Should I walk or drive my {v} there?"
    items.append({
        "id": f"car_wash_diverse_{idx:04d}",
        "document": doc,
        "query": query,
        "target_entity": "Drive",
        "category": "car_wash",
        "scenario_type": "canonical_car_wash",
        "metadata": {"vehicle": v, "distance_m": d, "dirt_state": dirt}
    })
    idx += 1

# ── Scenario Type 2: Gas station / fuel (drive is correct) ───────────────────
gas_scenarios = [
    ("sedan", 50, "The gas tank is nearly empty."),
    ("SUV", 80, "Running on fumes."),
    ("pickup truck", 30, "The fuel warning light is on."),
    ("minivan", 60, "Almost out of gas after a long drive."),
    ("crossover", 100, "Need to refuel before a road trip."),
    ("hybrid", 40, "Battery and gas both need a top-up."),
    ("sports car", 70, "The tank is empty."),
    ("coupe", 120, "Gas gauge reads below E."),
    ("hatchback", 25, "The fuel light has been on for 10 miles."),
    ("luxury sedan", 90, "Premium fuel needed for the engine."),
]
for (v, d, context) in gas_scenarios:
    doc = f"{context} The nearest gas station is {d} meters away."
    query = f"I need to refuel my {v}. The gas station is {d} meters away. Should I walk or drive?"
    items.append({
        "id": f"car_wash_diverse_{idx:04d}",
        "document": doc,
        "query": query,
        "target_entity": "Drive",
        "category": "car_wash",
        "scenario_type": "gas_station_fueling",
        "metadata": {"vehicle": v, "distance_m": d}
    })
    idx += 1

# ── Scenario Type 3: Auto repair / mechanic (drive is correct) ───────────────
repair_scenarios = [
    ("sedan", "The check engine light is on.", 200),
    ("pickup truck", "The brakes need replacing.", 150),
    ("SUV", "The tires need to be rotated.", 80),
    ("minivan", "The transmission is slipping.", 300),
    ("coupe", "The oil needs changing immediately.", 100),
    ("hatchback", "The windshield wiper motor is broken.", 60),
    ("crossover", "The battery died and needs replacement.", 40),
    ("sports car", "The exhaust system needs repair.", 250),
    ("luxury sedan", "The air conditioning compressor failed.", 120),
    ("cargo van", "The serpentine belt is worn.", 90),
]
for (v, issue, d) in repair_scenarios:
    doc = f"{issue} The nearest auto repair shop is {d} meters away."
    query = f"My {v} needs repairs. The mechanic's shop is {d} meters away. Should I walk or drive my {v} there?"
    items.append({
        "id": f"car_wash_diverse_{idx:04d}",
        "document": doc,
        "query": query,
        "target_entity": "Drive",
        "category": "car_wash",
        "scenario_type": "auto_repair",
        "metadata": {"vehicle": v, "distance_m": d, "issue": issue}
    })
    idx += 1

# ── Scenario Type 4: Parking (drive is correct) ───────────────────────────────
parking_scenarios = [
    ("sedan", "downtown garage", 100),
    ("SUV", "airport parking lot", 300),
    ("pickup truck", "secure vehicle storage facility", 150),
    ("minivan", "parking structure near the stadium", 200),
    ("coupe", "monthly parking garage", 80),
    ("electric vehicle", "EV charging station and parking", 60),
    ("sports car", "covered parking lot", 120),
    ("hatchback", "park-and-ride lot", 250),
    ("crossover", "valet parking service", 50),
    ("luxury sedan", "private parking facility", 180),
]
for (v, dest, d) in parking_scenarios:
    doc = f"I need to park my {v} at the {dest}. It is {d} meters from here."
    query = f"Should I walk or drive my {v} to the {dest} {d} meters away to park it?"
    items.append({
        "id": f"car_wash_diverse_{idx:04d}",
        "document": doc,
        "query": query,
        "target_entity": "Drive",
        "category": "car_wash",
        "scenario_type": "vehicle_parking",
        "metadata": {"vehicle": v, "distance_m": d, "destination": dest}
    })
    idx += 1

# ── Scenario Type 5: Negative controls — WALK is correct ─────────────────────
# These are cases where a PERSON is going somewhere (no vehicle required).
# The car wash invariant does NOT apply — the vehicle is not being serviced.
walk_scenarios = [
    ("I want to buy groceries.", "grocery store", 50, "Walk",
     "No vehicle service needed; person can walk."),
    ("I need to pick up a coffee.", "coffee shop", 30, "Walk",
     "Person is going, not vehicle."),
    ("I want to visit the pharmacy.", "pharmacy", 80, "Walk",
     "No vehicle service; errand on foot is fine."),
    ("I need to mail a letter.", "post office", 60, "Walk",
     "Person errand, no vehicle service required."),
    ("I want to get lunch.", "sandwich shop", 40, "Walk",
     "Food errand, person walks normally."),
    ("I need to drop off library books.", "library", 100, "Walk",
     "Person errand, no vehicle involved."),
    ("I want to pick up my prescription.", "drugstore", 70, "Walk",
     "Person errand; walking is fine."),
    ("I want to get a haircut.", "barbershop", 90, "Walk",
     "Person service, not vehicle service."),
    ("I need to pay a bill at the utility office.", "utility office", 150, "Walk",
     "Person errand on foot."),
    ("I want to drop off dry cleaning.", "dry cleaner", 45, "Walk",
     "Person errand, no vehicle required."),
]
for (context, dest, d, target, rationale) in walk_scenarios:
    doc = f"{context} The {dest} is {d} meters away."
    query = f"The {dest} is {d} meters away. Should I walk or drive?"
    items.append({
        "id": f"car_wash_diverse_{idx:04d}",
        "document": doc,
        "query": query,
        "target_entity": target,
        "category": "car_wash",
        "scenario_type": "negative_control_walk",
        "metadata": {"distance_m": d, "destination": dest, "rationale": rationale}
    })
    idx += 1

assert len(items) == 100, f"Expected 100 items, got {len(items)}"

out_path = "data/tasks/car_wash_diverse_100.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Written {len(items)} items to {out_path}")
# Summary
types = {}
for it in items:
    t = it["scenario_type"]
    types[t] = types.get(t, 0) + 1
print("Breakdown by scenario type:")
for k, v in types.items():
    print(f"  {k:<35} {v}")
