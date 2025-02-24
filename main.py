#!/usr/bin/env python3
import pandas as pd
import argparse
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_ingredients(sentence):
    """
    Extract ingredient words from the given sentence by removing punctuation,
    converting to lowercase, splitting on whitespace, and filtering out basic stopwords.
    
    :param sentence: A sentence describing the ingredients you want to use.
    :return: A string of extracted ingredient words separated by spaces.
    """
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    cleaned_sentence = sentence.translate(translator)
    # Convert to lowercase and split on whitespace
    tokens = cleaned_sentence.lower().split()
    # Define a simple set of stopwords to filter out common words
    stopwords = {"i", "want", "to", "use", "and", "or", "with", "the", "a", "an", "of", "in", "on", "for", "what", "ingredients"}
    ingredients = [token for token in tokens if token not in stopwords]
    return " ".join(ingredients)

def load_data(csv_path):
    """Load food data from a CSV file and clean column names."""
    df = pd.read_csv(csv_path)
    # Strip whitespace and convert column names to lowercase
    df.columns = df.columns.str.strip().str.lower()
    return df

def preprocess_data(df, text_column):
    """
    Preprocess the text data by filling missing values and ensuring string type.
    
    :param df: DataFrame containing the recipes.
    :param text_column: The name of the column with ingredient lists/descriptions.
    """
    df[text_column] = df[text_column].fillna("").astype(str)
    return df

def build_tfidf_matrix(text_data):
    """
    Build the TF-IDF matrix from the text data.
    
    :param text_data: A pandas Series or list of text data.
    :return: The TF-IDF matrix and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix, vectorizer

def get_recommendations(user_input, tfidf_matrix, vectorizer, df, top_n=5):
    """
    Compute cosine similarity between the user's input vector and each recipe's ingredients.
    
    :param user_input: The extracted ingredient words as a string.
    :param tfidf_matrix: TF-IDF matrix for the recipes.
    :param vectorizer: The fitted TF-IDF vectorizer.
    :param df: DataFrame containing the recipes.
    :param top_n: Number of top recommendations to return.
    :return: DataFrame of the top recommended recipes with their similarity scores.
    """
    user_vec = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices].copy()
    recommendations["similarity"] = cosine_sim[top_indices]
    return recommendations

def main():
    parser = argparse.ArgumentParser(
        description="Food Recipe Recommendation System (ingredient extraction from a sentence)"
    )
    parser.add_argument("--csv", type=str, default="food.csv", help="Path to the food CSV file")
    parser.add_argument("--sentence", type=str, help="A sentence describing the ingredients you want to use (e.g., 'I want to use chicken, garlic, and tomatoes')")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top recommendations to return")
    args = parser.parse_args()

    # Load and preprocess the data
    df = load_data(args.csv)
    # print("Columns in CSV:", df.columns.tolist())
    
    text_column = "ingredients"
    if text_column not in df.columns:
        print(f"Error: The CSV file does not contain a '{text_column}' column.")
        return
    df = preprocess_data(df, text_column)

    # Build TF-IDF matrix for the recipes' ingredient lists/descriptions
    tfidf_matrix, vectorizer = build_tfidf_matrix(df[text_column])

    # Get user input as a sentence, then extract ingredient keywords
    if args.sentence:
        user_sentence = args.sentence
    else:
        user_sentence = input("Enter what ingredients you want to use: ")

    # Extract ingredients from the sentence
    user_input = extract_ingredients(user_sentence)
    print(f"Extracted ingredients for recommendation: {user_input}")

    # Get recipe recommendations based on cosine similarity
    recommendations = get_recommendations(user_input, tfidf_matrix, vectorizer, df, top_n=args.top_n)

    # Determine the recipe name column (if available)
    recipe_col = None
    for col in ["title", "recipe", "name"]:
        if col in df.columns:
            recipe_col = col
            break

    print("\nTop Recommendations:")
    for idx, row in recommendations.iterrows():
        if recipe_col:
            recipe_name = row[recipe_col]
        else:
            recipe_name = row[text_column][:50] + "..."
        print(f"Recipe: {recipe_name} - Similarity Score: {row['similarity']:.2f}")

if __name__ == "__main__":
    main()
