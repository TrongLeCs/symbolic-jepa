# Symbolic-JEPA: Where Symbolic AI Meets Joint-Embedding Predictive Architecture for NL–FOL Conversion

---

## 🏗 Repository Structure

The project is organized into modular phases representing the training and evaluation lifecycle:

- [pretrain_phase/](pretrain_phase/): Stage 1 training using the JEPA architecture. Focuses on joint embedding of NL and FOL with a masking task and auxiliary structural losses.
- [finetune_prep_phase/](finetune_prep_phase/): Data preparation for the fine-tuning stage. Parses FOL into trees and generates structural labels (**CPP** and **LDP**).
- [finetune_phase/](finetune_phase/): Stage 2 training. Fine-tunes a T5 model with structural heads for NL-to-FOL translation using multi-task supervised learning.
- [metric_eval/](metric_eval/): Comprehensive evaluation suite to assess translation quality across syntax, semantics, and logic.

---

## 📦 Data Access

Pre-processed datasets for both pretraining and fine-tuning phases are available for download at the link below:

**[Download Processed Datasets (OneDrive)](https://1drv.ms/f/c/b75bec574f5e22fe/IgCH_qo1qS1mQagfL3yjHGTEAdFtIEmfq0_JSdqFRrksvBE?e=B7qQ2d)**

---

## 📊 Data Specifications

### 1. Pretraining Data (`pretrain_phase`)
The pretraining phase uses a rich JSONL format that includes path-level structural information for both NL and FOL.

*   **Format**: `.jsonl`
*   **Key Fields**:
    *   `topic`: A unique ID for the logic sample.
    *   `ast_fol`: Contains the FOL expression, its tokens, and **structural paths** (type paths and value paths) representing the logic tree hierarchy.
    *   `ast_nl`: Similar to `ast_fol`, but for the Natural Language sentence, aligning linguistic components with logic tree nodes.

*   **Preview Case**:
    ```json
    {
      "topic": 1,
      "ast_fol": [{
        "expression": "( FORALL x ( person(x) -> program(x) ) )",
        "tokens": [["(", 0], ["FORALL", 1], ...],
        "type_paths": [{"current_node": "FORALL", "paths": ["FORALL"], "path_ids": [1]}, ...],
        "value_paths": [...]
      }],
      "ast_nl": [...]
    }
    ```
    *For more details on data formats and complex examples, please refer to the `data/` or `output/` directories in each phase, such as `pretrain_phase/data/`.*

### 2. Fine-tuning Structural Supervision (`.npz`)
The `finetune_prep_phase` generates structural labels stored in Compressed NumPy files (`.npz`), which provide the necessary conditional structural awareness for the decoders.

#### **Compositional Path Prediction (CPP)** (`*_cpp_paths.npz`)
This leverages the **Structure-Aware Node & Path Encoder (SANE)** representation to encode the hierarchical path from root to leaf for every token in the FOL formula.
- `topic_ids`: (N,) - Mapping to the main dataset.
- `labels`: (N, L) - Target token IDs for the decoder.
- `cpp_paths`: (N, L, Depth) - The structural "coordinate" of each token in the logic tree based on the SANE architecture.
- `type_vocab_keys/vals`: Vocab mapping for structural node types (e.g., Predicate, Variable, Quantifier).

#### **Logical Dependency Prediction (LDP)** (`*_ldp_links.npz`)
This captures the logical dependencies and variable flow within the formula to enforce semantic constraints.
- `ldp_links`: (N, L, L) - Adjacency matrix of logical dependencies (Logic Data Paths).
- `ldp_edges`: (N, E, 2) - Explicit list of (source, destination) edges for variable binding (e.g. Predicate$\rightarrow$Argument edges).
- `tokens`: (N, L) - Tokens aligned with the Tokenizer pieces.
- `token_predicate_id`: (N, L) - Local IDs identifying tokens referring to specific logical predicates.

*   **Preview Case (Input JSON)**:
    ```json
    [
      {
        "topic": 1,
        "nl": "A person is considered a programmer if they can write computer code",
        "fol": "(FORALL x (person(x) AND can_write_code(x) IMPLIES program(x)))"
      }
    ]
    ```
    *For more details on how these records are paired with structural labels, check `finetune_phase/data/`.*

---

## ⚖️ Metrics & Evaluation

Evaluation consists of three complementary dimensions:

1.  **Well-formedness**: Validates if the generated FOL string is syntactically correct and parsable.
2.  **Semantic Score**: Measures linguistic similarity between the generated and reference FOL using metrics like **BLEU** or **BERTScore**.
3.  **Logic Score**: The most critical metric, assessing structural equivalence using FOL tree matching or SMT solvers to verify if the generated formula is logically identical to the ground truth.

---

## 🚀 Getting Started

### 1. Environment Setup
The project relies on two main Conda environments to separate training libraries from evaluation dependencies. 

Create and activate the environments (requires Python 3.10+):
```bash
# 1. Training Environment
conda create -n logic_jepa python=3.10
conda activate logic_jepa
pip install -r requirements.txt

# 2. Evaluation Environment (contains specific logic parsing and NLP dependencies)
conda create -n env_metric python=3.10
conda activate env_metric
pip install -r requirements.txt
```

### 2. Auto-Run the Full Pipeline
You can orchestrate the entire workflow (from pre-training to evaluation) using the master script:
```bash
bash script_run.sh
```

### 3. Running Phases Manually
If you want to run or modify specific parts of the architecture, navigate to each module:

#### Step 1: Data Preparation (Structural Labeling)
Generate structure labels (CPP and LDP) for fine-tuning.
```bash
conda activate logic_jepa
cd finetune_prep_phase
bash run_pipeline.sh --input data/sample.json --output_dir output/
```

#### Step 2: Pre-training (Symbolic-JEPA Encoder)
Pre-train the JEPA encoder to learn joint embeddings for NL and FOL.
```bash
cd pretrain_phase
python main.py
```

#### Step 3: Fine-tuning (Logic-Structured Decoder)
Fine-tune the T5 model using the pre-trained encoder and structural auxiliary heads.
```bash
cd finetune_phase
python main.py
```

#### Step 4: Inference
Generate FOL sequences from Natural Language.
```bash
cd finetune_phase
python inference.py --dataset_path data/test.json --preset B
```

#### Step 5: Evaluation
Evaluate the inference results against reference FOL using Syntactic, Semantic, and Logic scores.
```bash
conda activate env_metric
cd metric_eval
python main.py --input ../finetune_phase/inference_results/test.json --output results_metrics/test.json --lambda1 0.5 --lambda2 0.5
```