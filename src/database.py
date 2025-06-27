################################################################################
# Filename: database.py
#
# Purpose: 
# This script contains functions for handling database interactions, 
# including inserting, updating, and retrieving data.
#
# Author: Drew Heasman, P.Geo
# Last Updated: 27-06-2025
#
# Organization: University of Saskatchewan
################################################################################

################################################################################
# Future Improvements
################################################################################

# - Add an option to store logs in the database for better tracking.

################################################################################
# Libraries
################################################################################

# Standard Libraries
import os
import re
import csv
from io import StringIO
import logging
import pandas as pd
from sqlalchemy import text, create_engine
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from sqlalchemy import text

################################################################################
# Global Variables
################################################################################

TOTAL_TOKENS_PER_DOCUMENT = 4326  # Average number of tokens per document

# ✅ Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

################################################################################
# Database Connection - Cloud Ready
################################################################################

def get_database_engine():
    """
    Load PostgreSQL credentials from environment (.env file) and 
    return a SQLAlchemy engine to connect to cloud PostgreSQL.

    Supports both local and cloud database connections.
    """

    # Load from environment variables
    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")
    DB_PORT = os.getenv("DB_PORT", "5432")

    # Safety check
    if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
        raise ValueError("❌ Missing database credentials. Check your .env file!")

    # Build the full URL
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Create SQLAlchemy engine
    return create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)


################################################################################
# Insert and Update Database Functions
################################################################################

def process_and_import_data(root_folder, total_documents, engine, skip_processed=True, fast_insert=True):
    term_folders = os.listdir(root_folder)
    for term_folder in tqdm(term_folders, desc="Importing terms"):
        term_path = os.path.join(root_folder, term_folder)

        if skip_processed:
            with engine.connect() as connection:
                query = text("SELECT term FROM processed_terms WHERE term = :term")
                result = connection.execute(query, {"term": term_folder}).fetchone()
                if result:
                    continue

        processed_file = os.path.join(term_path, f"{term_folder}_processed_text.csv")
        if os.path.exists(processed_file):
            raw_df = pd.read_csv(processed_file)
            process_and_update_term_counts(raw_df, term_folder, total_documents, engine)

        stats_file = os.path.join(term_path, f"{term_folder}_cooccurrence_stats.csv")
        if os.path.exists(stats_file):
            stats_df = pd.read_csv(stats_file)
            stats_df.dropna(subset=["word_2"], inplace=True)
            stats_df = stats_df[stats_df["word_2"].str.strip() != ""]

            if fast_insert:
                try:
                    insert_cooccurrence_via_copy(engine, stats_df)
                except Exception as e:
                    print(f"⚠️ COPY failed for {term_folder}, trying row-by-row insert... Error: {e}")
                    insert_cooccurrence_to_postgres(engine, stats_df)
            else:
                insert_cooccurrence_to_postgres(engine, stats_df)

        with engine.connect() as connection:
            connection.execute(
                text("INSERT INTO processed_terms (term) VALUES (:term) ON CONFLICT (term) DO NOTHING"),
                {"term": term_folder}
            )
            connection.commit()

        logging.info(f"Finished processing term: {term_folder}")


def process_and_update_term_counts(raw_df, search_term, total_docs, engine):
    row_count = len(raw_df)
    total_tokens = total_docs * TOTAL_TOKENS_PER_DOCUMENT
    probability = row_count / total_tokens

    word_counts_data = {
        "word": search_term.lower(),
        "count": row_count,
        "probability": probability,
    }

    with engine.connect() as connection:
        query = text("""
        INSERT INTO term_counts (word, count, probability)
        VALUES (:word, :count, :probability)
        ON CONFLICT (word) DO UPDATE SET
            count = EXCLUDED.count,
            probability = EXCLUDED.probability;
        """)
        connection.execute(query, word_counts_data)
        connection.commit()

    print(f"Inserted or updated word_counts data for term: {search_term}")


def insert_cooccurrence_via_copy(engine, stats_df, max_retries=3, verbose=True):
    if stats_df.empty:
        print("⚠️ Skipping insert — co-occurrence DataFrame is empty.")
        return

    attempt = 0
    dropped_rows = []

    while attempt < max_retries:
        if verbose:
            print(f"\n🌀 COPY attempt {attempt + 1} — {len(stats_df)} rows")

        # Prepare in-memory CSV
        csv_buffer = StringIO()
        stats_df.to_csv(
            csv_buffer,
            sep=",",
            header=False,
            index=False,
            quoting=csv.QUOTE_ALL
        )
        csv_buffer.seek(0)

        raw_conn = engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            try:
                print(f"📤 Trying COPY insert (attempt {attempt + 1})...")
                cursor.copy_expert(
                    """
                    COPY term_cooccurrence (word_1, word_2, count, prob_w1w2)
                    FROM STDIN WITH CSV
                    """,
                    csv_buffer
                )
                raw_conn.commit()
                print("✅ COPY insert completed successfully.")
                return  # SUCCESS
            except Exception as e:
                raw_conn.rollback()
                print(f"❌ COPY failed on attempt {attempt + 1}: {e}")

                # Try to extract duplicate key info
                err_msg = str(e)
                match = re.search(r"\((word_1, word_2\)=\((.+?), (.+?)\)\)", err_msg)
                if match:
                    w1, w2 = match.group(2).strip(), match.group(3).strip()
                    print(f"⚠️ Problematic row: word_1='{w1}', word_2='{w2}' — removing and retrying")
                    dropped_rows.append((w1, w2))
                    stats_df = stats_df[~((stats_df["word_1"] == w1) & (stats_df["word_2"] == w2))]
                else:
                    print("⚠️ Could not identify offending row. Aborting retries.")
                    raise e
            finally:
                cursor.close()
        finally:
            raw_conn.close()

        attempt += 1

    print("❌ COPY insert failed after maximum retries.")
    if dropped_rows:
        print(f"🧹 Dropped {len(dropped_rows)} problematic row(s): {dropped_rows}")
    raise RuntimeError("COPY insert failed with too many conflicts.")


def insert_cooccurrence_to_postgres(engine, stats_df, batch_size=250):
    if stats_df.empty:
        print("⚠️ Co-occurrence DataFrame is empty. Skipping insert.")
        return

    query = text("""
        INSERT INTO term_cooccurrence (word_1, word_2, count, prob_w1w2)
        VALUES (:word_1, :word_2, :count, :prob_w1w2)
        ON CONFLICT (word_1, word_2) DO UPDATE SET
            count = EXCLUDED.count,
            prob_w1w2 = EXCLUDED.prob_w1w2;
    """)

    data = stats_df.to_dict(orient="records")
    total_rows = len(data)

    for i in tqdm(range(0, total_rows, batch_size), desc="   Inserting co-occurrence batches"):
        batch = data[i:i + batch_size]
        try:
            with engine.begin() as connection:
                connection.execute(query, batch)
        except Exception as e:
            print(f"❌ Failed to insert batch {i}-{i+batch_size}: {e}")

def remove_term(engine, term):
    term = term.strip().lower()

    queries = [
        text("DELETE FROM processed_terms WHERE term = :term"),
        text("DELETE FROM term_cooccurrence WHERE word_1 = :term OR word_2 = :term"),
        text("DELETE FROM term_counts WHERE word = :term")
    ]

    try:
        Session = sessionmaker(bind=engine)
        session = Session()

        for query in queries:
            session.execute(query, {"term": term})

        session.commit()
        print(f"✅ Successfully removed '{term}' from the database.")

    except Exception as e:
        session.rollback()
        print(f"❌ Error removing term '{term}': {e}")

    finally:
        session.close()
