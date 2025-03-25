# FactSpan Dataset and Expansion Scripts

This repository contains the FactSpan dataset, a collection of fact-checked claims gathered from Google Fact Check Explorer and annotated with ClaimAnnotation Markup. Additionally, it provides scripts to expand and update this dataset with new claims.

## Dataset Structure

The repository includes two primary datasets:

-   **FactSpan (data/FactSpan.csv):**
    -      A basic dataset containing scraped claims and their corresponding fact-check information.
    -      Designed for straightforward access and use in various natural language processing and fact-checking tasks.
-   **FactSpan_annotated (data/FactSpan_annotated.csv):**
    -      An enriched version of FactSpan with additional features extracted using a Large Language Model (LLM).
    -      Includes detailed annotations such as identified claim features (e.g., numerical data, quotes, etc.) and topic classification.
    -      Offers deeper insights into the claims' linguistic and contextual properties.

## Dataset Location

Both datasets are located in the `data/` directory of this repository.

## Expanding the Dataset

To facilitate dataset expansion, this repository provides a script called `update_dataset.py`, located in the `scripts/expansion/` directory. This script allows you to append new claims to an existing dataset.

### Usage

1.  **Navigate to the `scripts/expansion/` directory:**

    ```bash
    cd scripts/expansion/
    ```

2.  **Run the `update_dataset.py` script, providing the path to your new claims CSV file as an argument:**

    ```bash
    python update_dataset.py ../../data/newclaims_nomedia.csv
    ```

    -      Replace `../../data/newclaims_nomedia.csv` with the actual path to your CSV file containing the new claims.
    -   The script will append the new claims to the existing dataset.

### Requirements

-   Python 3.x
-   pandas library

### Notes

-   Ensure that the input CSV file (`newclaims_nomedia.csv` in the example) is formatted correctly and contains the necessary columns to be compatible with the existing dataset structure.
-   The `update_dataset.py` script assumes that the new claims dataset has the same columns as the original dataset.
-   The script will append the rows of the new csv to the end of the existing csv.
-   Please check the script to ensure it fits your use case, and modify it if needed.

## Contributing

Contributions to improve the dataset or the expansion scripts are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.