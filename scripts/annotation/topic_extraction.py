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
TOPIC_CATEGORIES = "(Health and Pandemics|Politics and Governance|Society and Culture|Economy and Environment|Conflict and Security)"
LOG_FILE = "logs/topic_annotation_log.txt"
OUTPUT_CSV_PREFIX = "logs/topic_extraction_"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(api_key=API_KEY)

def sample_data(data: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    unprocessed_data = data[~data['processed']]
    return unprocessed_data.sample(n=min(len(unprocessed_data), batch_size), random_state=1)

def get_model_responses(claims_batch: list, base_prompt: str, model: str) -> list:
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{timestamp} - Evaluating claims...")

    combined_claims = "\n".join([f"{i + 1}. {claim}" for i, claim in enumerate(claims_batch)])
    prompt = f"[INST]{base_prompt}\n{combined_claims} [/INST] Output: "

    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], max_tokens=1000, stop=["\n\n"], seed=1)
        message_content = response.choices[0].message.content
        logging.info(f"\nClaims: {combined_claims}\nMessage Content: {message_content}")
        # verdicts = [re.search(rf"{i}\.\s*{TOPIC_CATEGORIES}", message_content, re.IGNORECASE | re.DOTALL).group(1).strip() if re.search(rf"{i}\.\s*{TOPIC_CATEGORIES}", message_content, re.IGNORECASE | re.DOTALL) else 'Reevaluate' for i in range(1, len(claims_batch) + 1)]
        verdicts = [re.search(rf"{i}\.\s*(')?{TOPIC_CATEGORIES}(')?", message_content, re.IGNORECASE | re.DOTALL).group(
            2).strip() if re.search(rf"{i}\.\s*(')?{TOPIC_CATEGORIES}(')?", message_content,
                                    re.IGNORECASE | re.DOTALL) else 'Reevaluate' for i in
                    range(1, len(claims_batch) + 1)]
        logging.info(f"\n{verdicts}\n")
        duration = time.time() - start_time
        logging.info(f"{timestamp} - Evaluation completed in {duration:.2f} seconds.")
        # print(f"Evaluation took: {duration:.2f} seconds.")
        return verdicts
    except RateLimitError as e:
        logging.error(f"Rate limit error: {e}")
        time.sleep(10) #simple retry
        return get_model_responses(claims_batch, base_prompt, model)
    except APIError as e:
        logging.error(f"API error: {e}")
        return ['error'] * len(claims_batch)
    except Exception as e:
        logging.error(f"Error: {e}")
        return ['error'] * len(claims_batch)


def extract_topics(subset_data: pd.DataFrame, base_prompt: str, result_col: str, model: str) -> pd.DataFrame:
    for i in range(0, len(subset_data), BATCH_SIZE):
        batch_indexes = subset_data.index[i:i + BATCH_SIZE]
        logging.info(f"\n\n------\n\nBatch indices: {batch_indexes}\n")
        batch_claims = subset_data['claim'].loc[batch_indexes].tolist()
        batch_verdicts = get_model_responses(batch_claims, base_prompt, model)
        for idx, verdict in zip(batch_indexes, batch_verdicts):
            subset_data.at[idx, result_col] = verdict
    logging.info("Fact-check completed.")
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
            # print("Batch and progress saved.")
            logging.info("Batch and progress saved.")
        except Exception as e:
            logging.error(f"Error processing batch: {e}")


def annotate_dataset(data: pd.DataFrame):
    """Annotates claims in the DataFrame using OpenAI's GPT models."""

    result_col = 'topics'
    data['processed'] = False
    data[result_col] = pd.Series(dtype='object')
    base_prompt = """
    Classify each of the following claims into one of the following categories: 'Health and Pandemics', 'Politics and Governance', 'Society and Culture', 'Economy and Environment', or, 'Conflict and Security'. Please respond with the category name in English, even if the claim is in a non-English language. Analyze the claim's content and determine the most relevant category.
    Claims:
    """
    claims_left = data[~data['processed']].shape[0]
    print(f"Claims left to process: {claims_left}")
    logging.info(f"Claims left to process: {claims_left}")

    while claims_left > 0:
        control_batch_processing(data, base_prompt, result_col, total_batches=1, batch_size=150, process_func=extract_topics)
        reevaluate_count = data[data[result_col] == 'Reevaluate'].shape[0]
        print(f"Claims marked to be reevaluated: {reevaluate_count}")
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

if __name__ == "__main__":
    """Main execution block to annotate claims from a CSV file."""

    parser = argparse.ArgumentParser(description="Annotate claims using OpenAI.")
    parser.add_argument("input_csv", help="Input CSV file.")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.input_csv)
        annotate_dataset(data)
        data.to_csv(f'{args.input_csv.replace(".csv", "_annotated_t.csv")}', index=False)
        print(f"Annotation completed. Annotated data saved to {args.input_csv.replace('.csv', '_annotated_t.csv')}")
        logging.info(f"Annotation completed. Annotated data saved to {args.input_csv.replace('.csv', '_annotated_t.csv')}")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_csv}' not found.")
        logging.error(f"Error: Input file '{args.input_csv}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e}")