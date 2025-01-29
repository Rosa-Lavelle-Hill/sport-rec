import pandas as pd
import numpy as np
import re


def load_f1_scores(file_path):
    """
    Reads the GB model results from a text file and extracts the f1-scores per category.

    Args:
        file_path (str): Path to the text file containing GB model results.

    Returns:
        dict: A dictionary with category numbers as keys and f1-scores as values.
    """
    f1_scores = {}
    # Define a regex pattern to match lines with category numbers and f1-scores
    pattern = r'^\s*(\d+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+'

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(pattern, line)
                if match:
                    category = match.group(1)
                    f1_score = float(match.group(2))
                    f1_scores[category] = f1_score
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    # Ensure all 10 categories are present
    for cat in [str(i) for i in range(10)]:
        if cat not in f1_scores:
            f1_scores[cat] = 0.0  # Assign 0.0 if missing

    return f1_scores



def append_if_not_exists(filename, new_content):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            if new_content in content:
                print("Content already exists in the file.")
                return
    except FileNotFoundError:
        pass  # If the file doesn't exist, create it

    with open(filename, 'a', encoding='utf-8') as file:
        file.write(new_content + '\n')  # Append with a newline for clarity
    print("Content written to file.")
