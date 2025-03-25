import pandas as pd
import requests
import json
import sys
from datetime import datetime
import logging

import fact_checker_utils
import data_processing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def update_dataset(filepath):
    """Updates the dataset with new claims."""
    try:
        df = pd.read_csv(filepath)
        df['claimDate'] = pd.to_datetime(df['claimDate'], errors='coerce')  # handle bad dates
        today = datetime.now().date()
        latest_valid_date = df['claimDate'][df['claimDate'] < pd.to_datetime(today)].max().date()
        logging.info(f"Dataset last updated on: {latest_valid_date}")
    except FileNotFoundError:
        logging.error(f"File not found at {filepath}. Creating a new DataFrame.")
        df = pd.DataFrame(columns=['claim', 'claimDate', 'label', 'language', 'claim_year'])
        latest_valid_date = datetime(1970, 1, 1).date()

    url = "https://storage.googleapis.com/datacommons-feeds/claimreview/latest/data.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        claim_reviews_feed = data['dataFeedElement']
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return

    extracted_data = []
    for feed_item in claim_reviews_feed:
        if isinstance(feed_item, dict) and isinstance(feed_item.get('item'), list):
            for entry in feed_item['item']:
                if isinstance(entry, dict):
                    claim = entry.get('claimReviewed')
                    item_reviewed = entry.get('itemReviewed', {})

                    source = next((a.get('url') for a in item_reviewed.get('appearance', []) if isinstance(a, dict)), item_reviewed.get('firstAppearance', {}).get('url'))
                    if not source:
                        source = None

                    claim_date_str = next((a.get('datePublished') for a in item_reviewed.get('appearance', []) if isinstance(a, dict)), entry.get('datePublished'))

                    claim_date = None
                    if claim_date_str:
                        try:
                            claim_date = datetime.fromisoformat(claim_date_str.replace('Z', '+00:00')).date()
                        except ValueError:
                            logging.warning(f"Invalid date format: {claim_date_str}")

                    verdict = entry.get('reviewRating', {}).get('alternateName')

                    if claim and claim_date and latest_valid_date <= claim_date <= today:  # added claim check
                        extracted_data.append({
                            'claim': claim,
                            'claimDate': claim_date,
                            'source': source,
                            'verdict': verdict,
                            'raw_data': entry
                        })

    if extracted_data:
        new_df = pd.DataFrame(extracted_data)
        new_df['language'] = new_df['claim'].apply(data_processing.detect_language)
        # Load verdict mapping from JSON file
        try:
            with open('verdict_mapping.json', 'r') as f:
                verdict_mapping = json.load(f)
            new_df['label'] = new_df['verdict'].replace(verdict_mapping)
            new_df = new_df.drop('verdict', axis=1)  # remove verdict column
        except FileNotFoundError:
            logging.error("verdict_mapping.json not found.")

        # Load verdict prefix mapping
        try:
            with open('verdict_prefix_mapping.json', 'r') as f:
                verdict_prefix_mapping = json.load(f)

            def apply_prefix_mapping(label):
                for prefix, mapped_label in verdict_prefix_mapping.items():
                    if str(label).startswith(prefix):
                        return mapped_label
                return label  # Return original if no prefix match

            new_df['label'] = new_df['label'].apply(apply_prefix_mapping)

        except FileNotFoundError:
            logging.error("verdict_prefix_mapping.json not found.")

        # Identify non-standard labels
        non_standard_labels = \
        new_df[~new_df['label'].isin(['False', 'Partly False/Misleading', 'Mostly False', 'Mostly True', 'True'])][
            'label'].unique()
        logging.info(f"Non-standard labels in new_df: {non_standard_labels}")

        # Count rows before removal
        rows_before_removal = len(new_df)

        # Remove rows with non-standard labels
        new_df = new_df[
            new_df['label'].isin(['False', 'Partly False/Misleading', 'Mostly False', 'Mostly True', 'True'])]

        # Count rows after removal
        rows_after_removal = len(new_df)

        # Log the number of removed rows
        rows_removed = rows_before_removal - rows_after_removal
        logging.info(f"{rows_removed} rows were removed due to non-standard labels.")

        # Message to the user
        logging.info(
            "*It is recommended to inspect the non-standard labels and update the verdict_mapping.json "
            "or verdict_prefix_mapping.json files to include mappings for these values if appropriate.*"
        )

        new_df[['claim_reviewer_type', 'claim_reviewer_name', 'claim_reviewer_url']] = new_df['raw_data'].apply(data_processing.extract_author_data).apply(pd.Series)
        new_df['isValidFactChecker'] = new_df['claim_reviewer_url'].apply(fact_checker_utils.is_valid_fact_checker)

        # Filter only valid fact-checkers
        new_df = new_df[new_df['isValidFactChecker'] == True]
        logging.info(f"new_df has the columns: {new_df.columns}")
        logging.info(f"df has the columns: {df.columns}")
        logging.info(f"New df has {len(new_df)} rows.")
        logging.info(f"Old df had {len(df)} rows.")
        logging.info(f"Values in the column, label in df: {df['label'].unique()}")
        logging.info(
            f"Non-standard labels in new_df: {new_df[~new_df['label'].isin(['False', 'Partly False/Misleading', 'Mostly False', 'Mostly True', 'True'])]['label'].unique()}")
        df = pd.concat([df, new_df[['claim', 'claimDate', 'source', 'language', 'label', 'claim_reviewer_type', 'claim_reviewer_name', 'claim_reviewer_url', 'isValidFactChecker']]], ignore_index=True) # include source
        logging.info(f"After appending, df has {len(df)} rows.")
        new_df.to_csv('logs/temp_df.csv', index=False)
        # logging.info(f"Dataset updated successfully. {len(new_df)} new claims added.")

        # Filter only valid fact checkers from the entire dataframe
        # df = df[df['isValidFactChecker'] == True]
        # df.to_csv(filepath, index=False)
    else:
        logging.info("No new claims found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_dataset.py <filepath>")
    else:
        filepath = sys.argv[1]
        update_dataset(filepath)