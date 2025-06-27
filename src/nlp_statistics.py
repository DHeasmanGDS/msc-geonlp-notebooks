################################################################################
# Filename: nlp_statistics.py
#
# Purpose: 
# Core NLP statistics processing module for co-occurrence extraction and 
# corpus statistics generation from preprocessed geological text data.
#
# Author: Drew Heasman, P.Geo
# Last Updated: 13-06-2025
#
# Organization: University of Saskatchewan
################################################################################

################################################################################
# Future Improvements
################################################################################

# - Improve error handling when special characters or invalid folder names occur.
# - Optionally implement multiprocessing with safe deterministic output.
# - Optimize co-occurrence calculations for very large corpora.
# - Expand support for TF-IDF and similarity calculations if required.
# - Evaluate use of approximate nearest neighbors for large vector spaces.
# - Add global corpus-level statistics tracking.

################################################################################
# Libraries
################################################################################

# Standard Libraries
import os
import time
import math
import pandas as pd
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from annoy import AnnoyIndex
from sqlalchemy import text
import spacy
# Load SpaCy model once globally
nlp = spacy.load("en_core_web_sm")

from tqdm import tqdm

# Declare globally to avoid re-initializing every time
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

from logging_utils import write_statistics_log

################################################################################
# Global Variables
################################################################################

TOTAL_TOKENS_PER_DOCUMENT = 4326  # Average number of tokens per document (estimated)

################################################################################
# Term Processing Functions
################################################################################

def process_all_terms(root_folder, n, total_docs, overwrite=False):
    """
    Full pipeline to process all terms inside root_folder.

    This function iterates through each subfolder (each representing a search term)
    inside the provided root_folder. It calls process_single_term() for each term.

    Parameters:
    -----------
    root_folder : str
        Path to the folder containing all term subfolders.

    n : int
        Number of top co-occurring words to analyze for each term.

    total_docs : int
        Total number of documents (for probability normalization).

    overwrite : bool, optional (default=False)
        If True, reprocess terms even if co-occurrence CSV already exists.
    """
    term_folders = [
        f for f in os.listdir(root_folder) 
        if os.path.isdir(os.path.join(root_folder, f))
    ]

    for term_folder in term_folders:
        process_single_term(term_folder, root_folder, n, total_docs, overwrite)


def process_single_term(term_folder, root_folder, n, total_docs, overwrite=False):
    """
    Processes a single search term subfolder.

    This function reads the pre-processed text CSV for the term, computes
    co-occurrence statistics, generates corpus-level statistics, and saves 
    both outputs as CSV files.

    Parameters:
    -----------
    term_folder : str
        Name of the subfolder (search term being processed).

    root_folder : str
        Path to the parent folder containing all term subfolders.

    n : int
        Number of top co-occurring words to calculate.

    total_docs : int
        Total number of documents in the full dataset.

    overwrite : bool, optional (default=False)
        If True, overwrite existing CSV outputs.
    """
    term_path = os.path.join(root_folder, term_folder)
    stats_file = os.path.join(term_path, f"{term_folder}_cooccurrence_stats.csv")

    if not overwrite and os.path.exists(stats_file):
        print(f"✅ {term_folder} already processed. Skipping.")
        return

    processed_file = os.path.join(term_path, f"{term_folder}_processed_text.csv")
    if not os.path.exists(processed_file):
        print(f"⚠️ Processed file not found for {term_folder}. Skipping.")
        return

    print(f"\n📊 Processing: {term_folder}")
    df = pd.read_csv(processed_file)

    cooccurrence_df = calculate_cooccurrence(df, n, total_docs)
    cooccurrence_df.to_csv(stats_file, index=False)
    print(f"✅ Saved co-occurrence statistics to {stats_file}")

    # Generate per-term corpus-level summary:
    calculate_corpus_statistics(df, term_folder, cooccurrence_df, term_path)


################################################################################
# Normalization & Corpus Statistics Functions
################################################################################

def normalize_word(word):
    """
    Apply strict normalization to individual words:
    - Lowercase conversion.
    - Lemmatization using SpaCy.
    - Filters out very short words (<= 2 characters).
    
    Parameters:
    -----------
    word : str
        The input word to normalize.
    
    Returns:
    --------
    str or None
        Normalized word (lemma), or None if filtered out.
    """
    word = word.lower().strip()
    if len(word) <= 2:
        return None
    
    doc = nlp(word)
    lemma = doc[0].lemma_
    
    if len(lemma) <= 2:
        return None
    
    return lemma


def calculate_corpus_statistics(df, search_term, cooccurrence_df, term_path, log_folder="logs"):
    """
    Compute high-level corpus summary statistics for a specific search term, 
    write summary to CSV file, and append log entry for tracking.

    Statistics calculated include:
    - Total number of snippets.
    - Total number of unique documents (by DOI).
    - Approximate total tokens (based on co-occurrence counts).
    - Unique terms (co-occurrence count + search term).
    - Unique publishers.

    Parameters:
    -----------
    df : pandas.DataFrame
        The processed_text dataframe containing document metadata.

    search_term : str
        The search term associated with this term folder.

    cooccurrence_df : pandas.DataFrame
        The pre-computed co-occurrence dataframe for this term.

    term_path : str
        Path to save output CSV for this term.

    log_folder : str, optional
        Folder location to store log files (default = "logs").

    Returns:
    --------
    None
    """
    start_time = time.time()

    total_snippets = len(df)
    total_documents = df['doi'].nunique()
    total_tokens = cooccurrence_df['count'].sum() + total_snippets  # crude token approximation
    unique_terms = len(cooccurrence_df['word_2'].unique()) + 1  # add search term itself
    publisher_count = df['publisher'].nunique()

    summary_data = {
        'search_term': search_term,
        'total_snippets': total_snippets,
        'total_documents': total_documents,
        'total_tokens': total_tokens,
        'unique_terms': unique_terms,
        'unique_publishers': publisher_count,
    }

    stats_file = os.path.join(term_path, f"{search_term}_corpus_stats.csv")
    pd.DataFrame([summary_data]).to_csv(stats_file, index=False)
    print(f"📊 Saved corpus stats for {search_term} to {stats_file}")

    runtime = time.time() - start_time

    # Log the statistics as part of run tracking
    write_statistics_log(
        term=search_term,
        total_snippets=total_snippets,
        total_documents=total_documents,
        total_tokens=total_tokens,
        unique_terms=unique_terms,
        unique_publishers=publisher_count,
        runtime=runtime,
        log_folder=log_folder
    )
################################################################################
# Co-occurrence Functions
################################################################################

def calculate_cooccurrence(df, n, total_docs):
    """
    Calculate co-occurrence statistics for a single search term based on processed text data.

    This function performs:
    - Lemmatization of all words in the processed_text column.
    - Filtering of short words (length <= 2).
    - Removal of self-cooccurrence (i.e., the search term itself is excluded).
    - Frequency counting of co-occurring terms.
    - Stemming-based deduplication to merge similar morphological variants.
    - Computes approximate co-occurrence probabilities relative to total corpus size.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing processed text with at least columns:
        - 'processed_text' (tokenized and cleaned text as string)
        - 'search_term' (the associated search term)
    
    n : int
        Number of top co-occurring words to return (ranking by count).

    total_docs : int
        Total number of documents (used to estimate global token count for probability calculations).

    Returns:
    --------
    pandas.DataFrame
        Co-occurrence statistics with columns:
        - word_1: The search term.
        - word_2: Co-occurring word.
        - count: Frequency count of word_2.
        - prob_w1w2: Approximate co-occurrence probability relative to entire corpus.
    """
    
    total_tokens = total_docs * TOTAL_TOKENS_PER_DOCUMENT
    search_term = df['search_term'].unique()[0].lower()
    search_term_words = set(search_term.split())

    all_words = []

    # Efficient single-pass processing of all text rows
    for text in tqdm(df['processed_text'], desc=f"Building word list for '{search_term}'", unit=" docs"):
        words = text.split()

        # Strong lemmatization & filtering inside comprehension
        lemmatized_words = [
            lemmatizer.lemmatize(word.lower())
            for word in words
            if len(word) > 2  # ✅ Skip short words immediately
        ]

        all_words.extend(lemmatized_words)

    word_counts = Counter(all_words)

    # Remove self-cooccurrence: exclude search term words and plural forms
    for term_word in search_term_words:
        word_counts.pop(term_word, None)
        word_counts.pop(term_word + "s", None)

    top_words = word_counts.most_common(n)

    # Build dataframe and deduplicate similar forms via stemming
    data = [
        {
            'word_1': search_term,
            'word_2': word,
            'word_2_stemmed': stemmer.stem(word),
            'count': count,
            'prob_w1w2': count / total_tokens
        }
        for word, count in top_words
    ]

    cooccurrence_df = pd.DataFrame(data)

    # Deduplicate words that map to same stem, keeping highest count
    cooccurrence_df = cooccurrence_df.sort_values(by='count', ascending=False)
    cooccurrence_df = cooccurrence_df.drop_duplicates(subset=['word_1', 'word_2_stemmed'], keep='first')
    cooccurrence_df = cooccurrence_df.drop(columns=['word_2_stemmed'])

    return cooccurrence_df


################################################################################
# TF-IDF Functions - (Not integrated)
################################################################################

def process_and_save_tfidf(raw_df, search_term, total_docs, output_folder, overwrite=False):
    """
    Computes and saves Term Frequency (TF) and Term Frequency-Inverse Document Frequency (TF-IDF) values 
    for a given search term.

    This function processes raw snippet data to compute TF and TF-IDF values for a given `search_term`.
    The results are saved as a CSV file in the specified `output_folder`. If a TF-IDF file for the term
    already exists and `overwrite` is set to `False`, the function skips processing.

    Parameters:
    ----------
    raw_df : pandas.DataFrame
        DataFrame containing raw snippet data with expected columns:
        - `_gddid` (document ID)
        - `highlight` (text snippet from the document)

    search_term : str
        The search term for which TF and TF-IDF are calculated.

    total_docs : int
        The total number of documents in the dataset, used for IDF calculation.

    output_folder : str
        Path where the computed TF-IDF CSV file should be saved.

    overwrite : bool, optional (default: False)
        If `True`, recomputes and overwrites existing CSV files.
        If `False`, skips processing if the file already exists.

    Returns:
    -------
    None
        The function saves computed TF-IDF values to a CSV file and logs the process.

    Example Usage:
    --------------
    >>> df = pd.DataFrame({
    >>>     '_gddid': ['doc1', 'doc1', 'doc2'],
    >>>     'highlight': ['granite rock', 'granite formation', 'granite contains feldspar']
    >>> })
    >>> process_and_save_tfidf(df, "granite", total_docs=100000, output_folder="data/tfidf", overwrite=True)

    Features:
    ---------
    - **Aggregation of Snippets**: Merges all snippets per document to calculate TF.
    - **TF Calculation**: Computes term frequency based on the number of snippets per document.
    - **IDF Calculation**: Uses total document count to compute inverse document frequency.
    - **Optimized Storage**: Saves only relevant columns (`_gddid`, `tf`, `tfidf`) to reduce file size.

    Notes:
    ------
    - The function assumes `TOTAL_TOKENS_PER_DOCUMENT` is a global variable defining 
      the average number of tokens per document.
    - The TF-IDF formula used:
        ```
        idf = log(total_docs / (1 + doc_freq))
        tf = num_snippets / TOTAL_TOKENS_PER_DOCUMENT
        tfidf = tf * idf
        ```
    - The output file is saved as `<output_folder>/<search_term>_tfidf.csv`.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    tfidf_file = os.path.join(output_folder, f"{search_term}_tfidf.csv")

    if os.path.exists(tfidf_file) and not overwrite:
        print(f"TF-IDF already exists for {search_term}. Skipping.")
        return

    raw_df['_gddid'] = raw_df['_gddid'].astype(str).fillna('')
    raw_df['highlight'] = raw_df['highlight'].astype(str).fillna('')
    aggregated_text = raw_df.groupby('_gddid').agg({'highlight': ' '.join, '_gddid': 'size'}).rename(columns={'_gddid': 'num_snippets'}).reset_index()
    doc_freq = len(aggregated_text)
    idf = math.log(total_docs / (1 + doc_freq))
    aggregated_text['tf'] = aggregated_text['num_snippets'] / TOTAL_TOKENS_PER_DOCUMENT
    aggregated_text['tfidf'] = aggregated_text['tf'] * idf

    aggregated_text[['_gddid', 'tf', 'tfidf']].to_csv(tfidf_file, index=False)
    print(f"TF and TF-IDF values saved to: {tfidf_file}")

################################################################################
# Cosine Similarity Functions - (Not integrated)
################################################################################

def calculate_approximate_similarity(engine, tfidf_matrix, doc_index, n_trees=10, n_neighbors=10):
    """
    Computes approximate cosine similarity using Annoy and stores results in the database.

    This function builds an Annoy index for fast approximate nearest neighbor search 
    based on cosine similarity. The computed similarities are stored in a PostgreSQL 
    database, ensuring that precomputed similarities are not recalculated.

    Parameters:
    -----------
    engine : SQLAlchemy engine
        Database connection engine for PostgreSQL.

    tfidf_matrix : scipy.sparse.csr_matrix
        A sparse matrix representation of global TF-IDF values for documents.

    doc_index : list
        A list of `_gddid` values corresponding to the rows of `tfidf_matrix`.

    n_trees : int, optional (default: 10)
        The number of trees to build in the Annoy index. More trees increase accuracy
        but also slow down querying.

    n_neighbors : int, optional (default: 10)
        The number of nearest neighbors to retrieve for each document.

    Returns:
    --------
    None
        The function stores the computed similarities in the `similarity_results` table.

    Example Usage:
    --------------
    >>> from scipy.sparse import csr_matrix
    >>> import pandas as pd
    >>> tfidf_matrix = csr_matrix([[0.1, 0.3, 0.0], [0.0, 0.5, 0.2], [0.1, 0.0, 0.4]])
    >>> doc_index = ['doc1', 'doc2', 'doc3']
    >>> engine = get_database_engine()
    >>> calculate_approximate_similarity(engine, tfidf_matrix, doc_index, n_trees=10, n_neighbors=5)

    Features:
    ---------
    - **Annoy Index for Efficiency**: Uses `AnnoyIndex` with cosine distance for fast 
      nearest neighbor searches.
    - **Precomputed Similarity Check**: Avoids duplicate calculations by checking 
      existing results in the database.
    - **Efficient Bulk Processing**: Computes and inserts similarity values efficiently 
      while maintaining database integrity.

    Notes:
    ------
    - Annoy uses `angular` distance, which is equivalent to cosine similarity.
    - The similarity is computed as `1 - distance` to convert it back to cosine similarity.
    - Results are stored in the `similarity_results` table, with `ON CONFLICT` ensuring 
      existing values are updated.
    """
    
    n_docs, n_features = tfidf_matrix.shape
    annoy_index = AnnoyIndex(n_features, 'angular')

    # Build Annoy index
    print("Building Annoy index...")
    for i in range(n_docs):
        vector = tfidf_matrix[i].toarray().flatten()
        annoy_index.add_item(i, vector)

    annoy_index.build(n_trees)
    print("Annoy index built.")

    with engine.connect() as connection:
        for i, doc_id in enumerate(doc_index):
            # Check if document already has similarities computed
            query = text("SELECT 1 FROM similarity_results WHERE doc_id = :doc_id LIMIT 1")
            if connection.execute(query, {"doc_id": doc_id}).fetchone():
                print(f"Skipping document {doc_id}: Similarity already computed.")
                continue

            nearest_neighbors = annoy_index.get_nns_by_item(i, n_neighbors, include_distances=True)
            for neighbor_idx, distance in zip(*nearest_neighbors):
                similar_doc_id = doc_index[neighbor_idx]
                similarity = 1 - distance
                if doc_id != similar_doc_id:
                    similarity_query = text("""
                    INSERT INTO similarity_results (doc_id, similar_doc_id, similarity)
                    VALUES (:doc_id, :similar_doc_id, :similarity)
                    ON CONFLICT (doc_id, similar_doc_id) DO UPDATE SET
                        similarity = EXCLUDED.similarity;
                    """)
                    connection.execute(similarity_query, {
                        "doc_id": doc_id,
                        "similar_doc_id": similar_doc_id,
                        "similarity": similarity
                    })
        connection.commit()

    print("Similarity computation completed.")