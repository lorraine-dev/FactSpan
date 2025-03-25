import pandas as pd
import numpy as np
from openai import OpenAI, APIError, RateLimitError
from datetime import datetime
import time
import re
import argparse
import os
from dotenv import load_dotenv
import logging

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key not found.")

OPENAI_MODEL = "gpt-3.5-turbo-0125"
BATCH_SIZE = 15
LOG_FILE = "logs/feature_extraction_log.txt"
OUTPUT_CSV_PREFIX = "logs/feature_extraction_"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=API_KEY)

def sample_data(data: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    unprocessed_data = data[~data['processed']]
    return unprocessed_data.sample(n=min(len(unprocessed_data), batch_size), random_state=1)

def get_model_responses(claims_batch: list, base_prompt: str, model: str) -> list:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{timestamp} - Evaluating claims for feature extraction...")

    combined_claims = "\n".join([f"{i + 1}. {claim}" for i, claim in enumerate(claims_batch)])
    prompt = f"[INST]{base_prompt}\n{combined_claims} [/INST] Output: "

    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], max_tokens=1000, stop=["\n\n"], seed=1)
        message_content = response.choices[0].message.content
        logging.info(f"\nClaims: {combined_claims}\nMessage Content: {message_content}")
        verdicts = []
        for i in range(1, len(claims_batch) + 1):
            pattern = rf"{i}\.\s*(((Numerical Data|Entity/Event Properties|Position Statements|Quote|None)(,|\sand\s)?\s*)+)"
            match = re.search(pattern, message_content, re.IGNORECASE | re.DOTALL)
            if match:
                verdict_string = match.group(1)
                labels = re.split(r",\s*|\s+and\s+", verdict_string.strip())
                cleaned_labels = [label.strip() for label in labels if label.strip()]
                verdicts.append(', '.join(cleaned_labels))
            else:
                verdicts.append('Reevaluate')
                logging.info(f"\n{timestamp} - Reevaluation needed for claim {i} in the batch...\n")

        logging.info(f"\n{verdicts}\n")
        duration = time.time() - start_time
        logging.info(f"{timestamp} - Evaluation completed in {duration:.2f} seconds.")
        return verdicts
    except RateLimitError as e:
        logging.error(f"Rate limit error: {e}")
        time.sleep(10)
        return get_model_responses(claims_batch, base_prompt, model)
    except APIError as e:
        logging.error(f"API error: {e}")
        return ['error'] * len(claims_batch)
    except Exception as e:
        logging.error(f"Error: {e}")
        return ['error'] * len(claims_batch)


def extract_features(subset_data: pd.DataFrame, base_prompt: str, result_col: str, model: str) -> pd.DataFrame:
    for i in range(0, len(subset_data), BATCH_SIZE):
        batch_indexes = subset_data.index[i:i + BATCH_SIZE]
        logging.info(f"\n\n------\n\nBatch indices: {batch_indexes}\n")
        batch_claims = subset_data['claim'].loc[batch_indexes].tolist()
        batch_verdicts = get_model_responses(batch_claims, base_prompt, model)
        for idx, verdict in zip(batch_indexes, batch_verdicts):
            subset_data.at[idx, result_col] = verdict
    logging.info("Feature extraction completed.")
    return subset_data


def control_batch_processing(data: pd.DataFrame, base_prompt: str, result_col: str, total_batches: int, batch_size: int, process_func: callable):
    for i in range(total_batches):
        batch = sample_data(data, batch_size)
        logging.info(f"Batch {i + 1} size: {len(batch)}")
        try:
            batch = process_func(batch, base_prompt, result_col, OPENAI_MODEL)
            data.loc[batch.index, result_col] = batch[result_col]
            data.loc[batch.index, 'processed'] = True
            data.to_csv(f'{OUTPUT_CSV_PREFIX}{i + 1}.csv')
            logging.info("Batch and progress saved.")
        except Exception as e:
            logging.error(f"Error processing batch: {e}")


def annotate_features(data: pd.DataFrame):
    """Annotates claims in the DataFrame for feature extraction using OpenAI's GPT models."""

    result_col = 'gpt3.5 features'
    data['processed'] = False
    data[result_col] = pd.Series(dtype='object')
    base_prompt = """
    For each claim provided, identify and label the specific features it contains. The features to look for are: "Numerical Data", "Entity/Event Properties", "Position Statements", and "Quote". A claim may also have 'None' of these features. Use your pre-training knowledge to understand the claims based on their content and linguistic context. Here are examples to guide your judgment:

    Input:
    1. "People all over the world enjoy music." - This claim does not contain any of the specified features.
    2. "During his interview on March 3, 2021, the CEO stated, 'Our profits have doubled in the last two years due to our innovative approach.'" - This claim includes a "Quote" and "Numerical Data".
    3. "During the 2022 Climate Action Summit, the Canadian Minister of Environment declared that Canada will commit to zero emissions by 2050, aligning with the Paris Agreement goals" - This claim contains "Entity/Event Properties", and "Position Statements".

    Output:
    1. None
    2. Quote, Numerical Data
    3. Entity/Event Properties, Position Statements

    Claims:
    """
    claims_left = data[~data['processed']].shape[0]
    print(f"Claims left to process: {claims_left}")
    logging.info(f"Claims left to process: {claims_left}")

    while claims_left > 0:
        control_batch_processing(data, base_prompt, result_col, total_batches=1, batch_size=150, process_func=extract_features)
        reevaluate_count = data[data[result_col] == 'Reevaluate'].shape[0]
        print(f"Claims marked to be reevaluated: {reevaluate_count}")
        print(f"Claims left: {claims_left}")
        logging.info(f"Claims marked to be reevaluated: {reevaluate_count}")

        if reevaluate_count == 1 and claims_left == 1:  # handle special case.
            reevaluate_indices = data[data[result_col] == 'Reevaluate'].index
            data.loc[reevaluate_indices, result_col] = np.nan
            data.loc[reevaluate_indices, 'processed'] = False

            # Randomly select 2 more *processed* claims to reprocess
            processed_indices = data[data['processed']].index.tolist()
            if len(processed_indices) > 0:
                additional_indices = np.random.choice(processed_indices, min(2, len(processed_indices)), replace=False)
                data.loc[additional_indices, 'processed'] = False
                data.loc[additional_indices, result_col] = np.nan  # clear the previous result.


        elif reevaluate_count > 0:
            reevaluate_indices = data[data[result_col] == 'Reevaluate'].index
            data.loc[reevaluate_indices, result_col] = np.nan
            data.loc[reevaluate_indices, 'processed'] = False
        claims_left = data[~data['processed']].shape[0]
        print(f"Claims left to process: {claims_left}")
        logging.info(f"Claims left to process: {claims_left}")

    features_list = ['Position Statements', 'Entity/Event Properties', 'Quote', 'Numerical Data']
    for feature in features_list:
        data[feature] = data[result_col].apply(lambda x: 1 if feature in str(x) else 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate claims for features using OpenAI.")
    parser.add_argument("input_csv", help="Input CSV file.")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.input_csv)
        annotate_features(data)
        data.to_csv(f'{args.input_csv.replace(".csv", "_features_extracted.csv")}', index=False)
        print(f"Feature extraction completed. Annotated data saved to {args.input_csv.replace('.csv', '_features_extracted.csv')}")
        logging.info(f"Feature extraction completed. Annotated data saved to {args.input_csv.replace('.csv', '_features_extracted.csv')}")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_csv}' not found.")
        logging.error(f"Error: Input file '{args.input_csv}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e}")