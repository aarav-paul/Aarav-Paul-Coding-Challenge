# Food Recipe Recommendation System

This repository contains a simple content-based recommendation system that suggests food recipes based on the ingredients provided by a user. The system uses TF-IDF vectorization and cosine similarity to compare a user's input (extracted from a sentence) with a dataset of recipes stored in a CSV file.

## Overview

The system:
- **Loads** a CSV file (`food.csv`) containing recipes.
- **Prompts** the user to enter a sentence describing the ingredients they want to use.
- **Extracts** key ingredient words from the sentence using basic Python string methods.
- **Computes** TF-IDF vectors for the recipes’ ingredient lists.
- **Calculates** cosine similarity between the user's ingredients and each recipe.
- **Returns** the top recommended recipes along with their similarity scores.

## Dataset

The system expects a CSV file named `food.csv` with columns including (but not limited to):
- `Title`: The name of the recipe.
- `Ingredients`: A text field containing the list or description of ingredients.
- `Instructions`: (Optional) Cooking instructions.
- Other columns (e.g., `Image_Name`, `Cleaned_Ingredients`) may be present.

**Note:** The system relies on the CSV having an `ingredients` column (case-insensitive). If your dataset uses a different name, please adjust the code accordingly.

## Requirements

- Python 3.6+
- Python packages:
  - `pandas`
  - `scikit-learn`

You can install the required packages using:

```bash
pip install -r requirements.txt
