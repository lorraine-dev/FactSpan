# FactSpan Dataset Expansion

This repository contains the FactSpan dataset, an extension of the X-Fact dataset, designed to support multilingual fact-checking research. It includes tools to expand and update the dataset with recent claims from the ClaimReview Markup for Data Commons Feed.

## Dataset Overview

The FactSpan dataset addresses limitations in existing multilingual fact-checking datasets by incorporating recent data and providing detailed annotations. The dataset includes:

-   Claims from both the X-Fact dataset (up to 2020) and the Data Commons Feed (post-2020).
-   Claims filtered from organizations recognized by the International Fact-Checking Network (IFCN) and Duke Reporters’ Lab, ensuring high reliability.
-   Standardized verdict labels (False, Mostly False, Partly False/Misleading, Mostly True, True) for consistency.
-   Rich annotations:
    - Topic Extraction (Health and Pandemics, Politics and Governance, etc.)
    - Claim Type Identification (factual or opinion)
    - Additional key features (Numerical Claims, Quotes, Position Statements, Entity/Event Properties)

## Repository Structure
````
.
├── Data
│   ├── FactSpan.csv               # Original FactSpan dataset.
│   ├── FactSpan_annotated.csv     # FactSpan dataset with annotations.
│   └── ValidFactCheckers          # Lists of valid fact-checking organizations.
│       ├── duke_list.txt
│       └── ifcn_list.txt
├── LICENSE
├── README.md
├── requirements.txt
└── scripts
    ├── annotation                 # Scripts for annotating the dataset.
    │   ├── annotate_factspan.py  # Main annotation script.
    │   ├── topic_extraction.py
    │   ├── claim_type_extraction.py
    │   ├── feature_extraction.py
    │   └── logs                  # Annotation logs.
    └── expansion                  # Scripts for expanding the dataset.
        ├── data_processing.py    # Data processing utilities.
        ├── fact_checker_utils.py # Fact-checker validation utilities.
        ├── update_dataset.py     # Script to update the dataset.
        ├── verdict_mapping.json  # Mapping of verdict labels.
        └── verdict_prefix_mapping.json # Mapping of verdict prefix labels.
````
## Dataset Update Instructions

### Option 1: Expanding the Unannotated Dataset (FactSpan.csv)

To expand the `FactSpan.csv` dataset with new claims, follow these steps:

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd FactSpan
    ```
2.  Navigate to the `scripts/expansion` directory:
    ```bash
    cd scripts/expansion
    ```
3.  Run the `update_dataset.py` script, specifying the path to `FactSpan.csv`:
    ```bash
    python update_dataset.py ../../Data/FactSpan.csv
    ```

This will update the `FactSpan.csv` file with the latest claims from the Data Commons Feed.

### Option 2: Expanding the Annotated Dataset (FactSpan_annotated.csv)

To expand the `FactSpan_annotated.csv` dataset, which requires LLM-based annotations, you need to configure your OpenAI API token:

1.  Add your OpenAI API token to a `.env` file in the root directory. Create the `.env` file if it doesn't exist. Add the following line to the file, replacing `<your_openai_token>` with your actual token:
    ```
    OPENAI_API_KEY=<your_openai_token>
    ```
2.  Clone the repository and navigate to the `scripts/expansion` directory (as in Option 1).
3.  Run the `update_dataset.py` script with the `--annotations` flag, specifying the path to `FactSpan_annotated.csv`:
    ```bash
    python update_dataset.py ../../Data/FactSpan_annotated.csv --annotations
    ```

This will update the `FactSpan_annotated.csv` file with new claims and their corresponding annotations.

**Note:** The `.env` file and the `scripts/annotation/logs` and `scripts/expansion/logs` directories are ignored by Git, as specified in the `.gitignore` file.