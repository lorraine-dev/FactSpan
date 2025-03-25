import pandas as pd
import subprocess
import os
import argparse
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def annotate_dataset(input_csv, output_dir):
    """
    Annotates a dataset by running feature_extraction, topic_extraction, and claim_type_extraction scripts.

    Args:
        input_csv (str): Path to the input CSV file.
        output_dir (str): Directory where the annotated CSV should be saved.
    """
    try:
        # Feature Extraction
        logging.info(f"Running feature_extraction.py on {input_csv}")
        subprocess.run(['python', 'feature_extraction.py', input_csv], check=True, cwd=os.path.dirname(os.path.abspath(__file__))) #run from current directory
        feature_extracted_csv = input_csv.replace('.csv', '_features_extracted.csv')
        logging.info(f"Feature extraction completed. Output: {feature_extracted_csv}")

        # Topic Extraction
        logging.info(f"Running topic_extraction.py on {feature_extracted_csv}")
        subprocess.run(['python', 'topic_extraction.py', feature_extracted_csv], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        topic_extracted_csv = feature_extracted_csv.replace('.csv', '_annotated_t.csv')
        logging.info(f"Topic extraction completed. Output: {topic_extracted_csv}")

        # Claim Type Extraction
        logging.info(f"Running claim_type_extraction.py on {topic_extracted_csv}")
        subprocess.run(['python', 'claim_type_extraction.py', topic_extracted_csv], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        claim_type_extracted_csv = topic_extracted_csv.replace('.csv', '_annotated_ct.csv')
        logging.info(f"Claim type extraction completed. Output: {claim_type_extracted_csv}")

        # Mapped Label Processing and claim_year extraction
        logging.info(f"Processing mapped_label and claim_year for {claim_type_extracted_csv}")
        data = pd.read_csv(claim_type_extracted_csv)

        # Extract claim_year
        data['claim_year'] = data['claimDate'].apply(
            lambda x: datetime.fromisoformat(x.replace('Z', '+00:00')).year if pd.notna(x) else None)

        data['mapped_label'] = data['label'].apply(lambda x: 'True' if x in ['Mostly True', 'True'] else 'False')
        final_output_csv = input_csv.replace('.csv', '_annotated.csv') #save the output to the same location as the input.
        data = data.drop(columns=['processed'], errors='ignore')  # remove processed column.
        data.to_csv(final_output_csv, index=False)
        logging.info(f"Mapped label processing completed. Final output: {final_output_csv}")

        # Clean up intermediate files
        os.remove(feature_extracted_csv)
        os.remove(topic_extracted_csv)
        os.remove(claim_type_extracted_csv)
        logging.info("Intermediate files removed.")

        print(f"Annotation process completed. Annotated data saved to {final_output_csv}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing annotation script: {e}")
        print(f"Error executing annotation script: {e}")
    except FileNotFoundError:
        logging.error(f"Error: Input file '{input_csv}' not found.")
        print(f"Error: Input file '{input_csv}' not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate a dataset using annotation scripts.")
    parser.add_argument("input_csv", help="Input CSV file to annotate.")
    args = parser.parse_args()

    # Extract the directory of the input CSV to store the output in the same location
    input_dir = os.path.dirname(os.path.abspath(args.input_csv))

    annotate_dataset(args.input_csv, output_dir=input_dir)