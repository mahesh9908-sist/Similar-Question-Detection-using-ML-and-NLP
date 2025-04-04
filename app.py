import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset Path
DATASET_PATH = "questions.csv"

# Load Dataset
def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        if "cleaned_text" not in df.columns:
            df["cleaned_text"] = df["question"].apply(preprocess_text)
        st.success("✅ Dataset loaded successfully!")
    except pd.errors.ParserError as e:
        st.error(f"❌ Dataset contains formatting errors: {e}")
        df = pd.DataFrame(columns=["question", "cleaned_text"])
    except FileNotFoundError:
        st.warning("Dataset not found. A new dataset will be created upon saving.")
        df = pd.DataFrame(columns=["question", "cleaned_text"])
    except Exception as e:
        st.error(f"❌ Failed to load the dataset: {e}")
        df = pd.DataFrame(columns=["question", "cleaned_text"])
    return df


# Save Dataset
def save_dataset(df):
    try:
        df.to_csv(DATASET_PATH, index=False)
        st.success("✅ Question saved successfully!")
    except PermissionError:
        st.error("❌ Failed to save the question to dataset. Please close 'questions.csv' if it is open elsewhere.")
    except Exception as e:
        st.error(f"❌ Failed to save the question to dataset: {e}")


# Preprocess Input
def preprocess_text(text):
    return text.strip().lower()


# Search for Similar Questions
def find_similar_questions(user_input, df):
    # Preprocess user input
    cleaned_input = preprocess_text(user_input)

    # Check if dataset is empty
    if df.empty:
        st.warning("The dataset is empty. No similar questions found.")
        return

    # Vectorize using TFIDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_text"].values.tolist() + [cleaned_input])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    threshold = 0.5  # Similarity threshold
    similar_indices = [i for i, score in enumerate(similarity_scores[0]) if score >= threshold]

    if similar_indices:
        st.write("### Similar Questions Found:")
        for index in similar_indices:
            st.write(f"**Question:** {df.iloc[index]['question']}")
            st.write(f"**Similarity Score:** {similarity_scores[0][index]:.2f}")
    else:
        st.write("No similar questions found.")


# Main App
st.title("Similar Question Detection Through ML and NLP")
st.write("This app detects questions in the dataset that are similar to the input question. If no similar question is found, you can save the new question to the dataset.")

# Load Dataset
df = load_dataset()

# Input Section
st.header("Enter a question:")
user_input = st.text_input("Type a question here:")

# Search Button
if st.button("Search for Similar Questions"):
    if user_input:
        find_similar_questions(user_input, df)
    else:
        st.warning("Please enter a question to search.")

# Save Button
st.subheader("Save question to dataset")
if st.button("Save Question to Dataset"):
    if user_input:
        cleaned_input = preprocess_text(user_input)
        if df.empty or cleaned_input not in df["cleaned_text"].values:
            new_row = pd.DataFrame([{"question": user_input, "cleaned_text": cleaned_input}])
            df = pd.concat([df, new_row], ignore_index=True)
            save_dataset(df)
        else:
            st.info("✅ The question already exists in the dataset.")
    else:
        st.warning("Please enter a question to save.")

