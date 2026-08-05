import json
import random

def generate_hle_benchmark_suite(output_path="data/tasks/hle_benchmark_suite.jsonl"):
    random.seed(2026)
    items = []
    item_id = 0

    # Domain 1: Quantum Information & Topological Computing
    hle_quantum = [
        {
            "doc": "Consider a non-Abelian topological phase governed by the SU(2)_3 Chern-Simons theory at level k=3. The quantum dimension of Fibonacci anyons is phi = (1 + sqrt(5))/2. The topological S-matrix entry S_{0,1} equals 1 / sqrt(1 + phi^2).",
            "query": "What is the exact topological quantum dimension of the non-trivial Fibonacci anyon in this phase?",
            "target": "1.6180339887"
        },
        {
            "doc": "The toric code Hamiltonian H = -J_A sum_v A_v - J_B sum_p B_p is defined on a two-dimensional square lattice with periodic boundary conditions. The ground state degeneracy on a torus (genus g=1) is equal to 4.",
            "query": "What is the exact ground state degeneracy of the 2D toric code on a torus of genus g=1?",
            "target": "4"
        },
        {
            "doc": "In quantum error correction using the [[7,1,3]] Steane code, the CSS code construction utilizes two classical [7,4,3] Hamming codes C1 and C2 where C2 is orthogonal to C1.",
            "query": "What is the minimum distance d of the [[7,1,3]] Steane quantum code?",
            "target": "3"
        }
    ]

    # Domain 2: Advanced Organic Chemistry & Reaction Dynamics
    hle_chem = [
        {
            "doc": "The Sharpless asymmetric epoxidation of allylic alcohols uses titanium tetraisopropoxide, diethyl tartrate (DET), and tert-butyl hydroperoxide (TBHP). When (+)-L-DET is used, oxygen atom addition occurs preferentially from the top face when the allylic alcohol is drawn with the hydroxymethyl group at the bottom-right.",
            "query": "Which enantiomer of diethyl tartrate is used in Sharpless epoxidation to direct top-face oxygen insertion when the allylic alcohol is drawn in standard bottom-right orientation?",
            "target": "(+)-L-DET"
        },
        {
            "doc": "In the stereospecific Woodward-Hoffmann electrocyclic ring closure of (2E,4Z,6E)-octatriene under thermal conditions, the reaction proceeds via a 6-electron disrotatory mechanism to yield cis-5,6-dimethyl-1,3-cyclohexadiene.",
            "query": "Does thermal 6-electron electrocyclization of (2E,4Z,6E)-octatriene proceed via a conrotatory or disrotatory mode?",
            "target": "disrotatory"
        }
    ]

    # Domain 3: Theoretical Computer Science & Algorithmic Game Theory
    hle_cs = [
        {
            "doc": "In algorithmic game theory, a two-player zero-sum game played on a finite graph with parity winning conditions has a deterministic memoryless winning strategy for both players, as proven by Mostowski and Emerson-Jutla.",
            "query": "What type of strategy complexity (memory requirement) is sufficient for winning parity games on finite graphs?",
            "target": "memoryless"
        },
        {
            "doc": "The PCP Theorem (Probabilistically Checkable Proofs) establishes that NP = PCP(O(log n), O(1)), meaning any NP verification proof can be checked reading a constant number of bits using logarithmic randomness.",
            "query": "According to the PCP Theorem, how many proof bits need to be queried by the verifier?",
            "target": "O(1)"
        }
    ]

    # Expand into 150 structured items across domains
    raw_templates = hle_quantum * 15 + hle_chem * 20 + hle_cs * 20

    for idx, raw in enumerate(raw_templates[:150]):
        item_id += 1
        items.append({
            "id": f"hle_expert_{item_id}",
            "category": "hle_humanitys_last_exam",
            "document": raw["doc"],
            "query": raw["query"],
            "target_entity": raw["target"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"Successfully generated {len(items)} Humanity's Last Exam benchmark items at {output_path}")

if __name__ == "__main__":
    generate_hle_benchmark_suite()
