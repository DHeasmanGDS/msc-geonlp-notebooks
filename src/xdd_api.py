################################################################################
# Filename: xdd_api.py
#
# Purpose: 
# This script contains functions designed to extract and process data 
# from the xDD API, including retrieving scientific snippets, handling API 
# requests, and managing parallel processing.
#
# Author: Drew Heasman, P.Geo
# Last Updated: 26-06-2025
#
# Organization: University of Saskatchewan
#
# SPDX-License-Identifier: MIT
# Data note: Outputs derived from xDD content are subject to xDD’s CC BY-NC 4.0 license.
################################################################################

################################################################################
# Future Improvements
################################################################################

# - Incorporate better logging files using logging_utils.py
# - Add support for asynchronous API requests using `asyncio` and `aiohttp` 
#   for better performance in high-volume queries.
# - Improve logging to include detailed error reporting and timestamps.
# - Add an option to process terms from an external text or CSV file.

################################################################################
# Libraries
################################################################################

# Standard Libraries
import os
import sys
import time
import requests
import pandas as pd
import threading
import csv
import json
import math
import re
from datetime import datetime


# Ensure the 'src' directory is included in the import path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

# Internal Module Imports
from text_processing import preprocess_text
from logging_utils import write_log_for_term

################################################################################
# Global Variables
################################################################################

# XDD API url endpoint
API = "https://xdd.wisc.edu/api/v1/snippets"

################################################################################
# Main Function Call
################################################################################

def process_and_save_results(search_terms, params, headers, root_folder, overwrite=False):
    """
    Sequentially processes search terms, saves results, and provides periodic status updates.
    After each term is processed (or if an existing CSV is detected), automatically builds:
      - {term}.bibjson (one record per unique document)
      - {term}_bib_map.csv (snippet row -> bibjson_id mapping)
    """

    all_dfs = []
    start_time = time.time()

    # Start periodic status update in a separate thread
    status_thread = threading.Thread(target=periodic_status_update, args=(start_time, search_terms), daemon=True)
    status_thread.start()

    for idx, term in enumerate(search_terms, 1):
        print(f"[INFO] ({idx}/{len(search_terms)}) Processing term '{term}'...")
        try:
            result = process_term(term, params, headers, root_folder, overwrite)
            if result is not None:
                all_dfs.append(result)
                print(f"[INFO] Completed '{term}' successfully.")
            else:
                print(f"[INFO] No data retrieved or processing skipped for '{term}'.")
        except Exception as e:
            print(f"[ERROR] Processing term '{term}' failed: {e}")

        # --- Always attempt to build Bibliography from the snippet CSV if present ---
        try:
            term_l = term.lower()
            term_folder = os.path.join(root_folder, term_l)
            csv_path = os.path.join(term_folder, f"{term_l}_processed_text.csv")
            out_bib = os.path.join(term_folder, f"{term_l}.bibjson")
            out_map = os.path.join(term_folder, f"{term_l}_bib_map.csv")

            if os.path.exists(csv_path):
                print(f"[BIBJSON] Building bibliography for '{term}' from {csv_path} ...")
                build_bibjson_from_snippets_csv(
                    csv_path=csv_path,
                    out_bibjson_path=out_bib,
                    out_map_csv=out_map,
                    collection_name=f"xDD-{term_l}",
                    chunksize=100_000,
                    title_year_fallback=True
                )
            else:
                print(f"[BIBJSON] Skipped for '{term}': processed_text CSV not found at {csv_path}")
        except Exception as e:
            print(f"[BIBJSON] Failed for '{term}': {e}")

    total_runtime = time.time() - start_time
    print(f"[COMPLETED] Processed {len(all_dfs)}/{len(search_terms)} terms in {total_runtime / 60:.2f} minutes.")

    return all_dfs


################################################################################
# Status Update Thread
################################################################################

def periodic_status_update(start_time, search_terms):
    """
    Prints a status update exactly every 2 hours to ensure the script is still running.

    Parameters:
    -----------
    start_time : float
        The timestamp when the script started.
    search_terms : list
        List of terms being processed.
    """
    while True:
        elapsed_time = time.time() - start_time
        print(f"[STATUS UPDATE] Elapsed Time: {elapsed_time / 3600:.2f} hours - Processing {len(search_terms)} terms...")
        time.sleep(2 * 3600)  # Sleep for exactly 2 hours before printing again

################################################################################
# Functions
################################################################################

def xdd_api_call(term, params, headers, retry_limit=300):
    """
    Makes an API request to xDD to fetch snippets related to the given term.
    Handles errors, retries upon failure, and supports pagination.

    Parameters:
    -----------
    term : str
        The search term to query in the xDD API.
    params : dict
        Dictionary containing query parameters for the API request.
    headers : dict
        Dictionary containing headers (e.g., API keys) for authentication.
    retry_limit : int, optional
        The maximum number of retries if the API request fails (default: 300).

    Returns:
    --------
    docs : list
        A list of retrieved documents/snippets from the API.
    runtime : float
        Time taken (in seconds) for the API request.

    Example Usage:
    --------------
    >>> params = {"api_key": "my_api_key", "limit": 100}
    >>> headers = {"Authorization": "Bearer my_token"}
    >>> documents, request_time = xdd_api_call("volcanic arc", params, headers)

    Features:
    ---------
    - **Automatic Pagination**: If additional pages of results exist, the function follows the `next_page` link.
    - **Robust Error Handling**:
      - Retries on API failures (timeouts, connection issues, non-200 responses).
      - Prints an error message if the maximum retry limit is reached.
    - **Dynamic Backoff**: Introduces a 5-second wait before retrying to reduce API overload.
    - **Performance Logging**: Returns runtime to help measure API response time.

    Notes:
    ------
    - `API` should be defined globally or replaced with the actual endpoint.
    - Ensure the `params` dictionary includes the required fields (`api_key`, `term`, etc.).
    - Exceeding the API rate limit may require adjusting the `retry_limit` or increasing the backoff time.
    """
    
    api_url = API
    params['term'] = term
    docs = []
    retries = 0
    start_time = time.time()

    while True:
        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                docs += data['success']['data']
                if 'next_page' in data['success'] and data['success']['next_page']:
                    api_url = data['success']['next_page']
                else:
                    break
            else:
                retries += 1
                if retries >= retry_limit:
                    print(f"Exceeded retry limit for term: {term}")
                    break
                time.sleep(5)
        except requests.exceptions.RequestException as req_err:
            retries += 1
            if retries >= retry_limit:
                print(f"Request error for term {term}: {req_err}")
                break
            time.sleep(5)
        except Exception as e:
            retries += 1
            if retries >= retry_limit:
                print(f"Unexpected error for term {term}: {e}")
                break
            time.sleep(5)

    runtime = time.time() - start_time
    return docs, runtime


def process_term(term, params, headers, root_folder, overwrite=False):
    """
    Processes a single search term by fetching data, saving results, and logging.
    
    This function:
    - Ensures the output directory exists.
    - Optionally skips processing if the term has already been processed, unless 'overwrite=True'.
    - Fetches data from the xDD API.
    - Processes text by extracting and cleaning highlights.
    - Saves results as a CSV file.
    - Logs execution details, including API runtime and document count.

    Parameters:
    -----------
    term : str
        The search term to process.
    params : dict
        API request parameters.
    headers : dict
        Headers for the API request (e.g., authentication token).
    root_folder : str
        The directory where processed results should be saved.
    overwrite : bool, optional
        Whether to overwrite existing processed files (default: False).

    Returns:
    --------
    pd.DataFrame or None:
        A Pandas DataFrame containing processed snippets if successful, otherwise None.

    Example Usage:
    --------------
    >>> params = {"api_key": "my_api_key", "limit": 100}
    >>> headers = {"Authorization": "Bearer my_token"}
    >>> root_folder = "data/results"
    >>> df = process_term("volcanic arc", params, headers, root_folder)

    Features:
    ---------
    - **Prevents Duplicate Processing**: Checks for existing processed files before running.
    - **Automated Data Fetching**: Calls `xdd_api_call()` to retrieve snippets from the xDD API.
    - **Text Preprocessing**: Cleans extracted text using `preprocess_text()`.
    - **Structured Storage**: Saves processed text in `root_folder/{term}/{term}_processed_text.csv`.
    - **Detailed Logging**: Calls `write_log_for_term()` to track execution time and document count.

    Notes:
    ------
    - The function assumes that `xdd_api_call()`, `preprocess_text()`, and `write_log_for_term()` are properly defined.
    - If the API does not return any results, the function returns `None`.
    """
    
    term = term.lower()
    term_subfolder = os.path.join(root_folder, term)

    # Ensure the folder exists
    os.makedirs(term_subfolder, exist_ok=True)

    processed_text_filename = os.path.join(term_subfolder, f"{term}_processed_text.csv")

    # Skip processing if the file already exists
    if os.path.exists(processed_text_filename) and not overwrite:
        print(f"Skipping term '{term}' as processed file already exists. Use overwrite=True to reprocess.")
        return None

    print(f"Processing term '{term}'...")  # Start message

    # Fetch data from xDD API
    docs, runtime = xdd_api_call(term, params, headers)

    if docs:
        df = pd.DataFrame(docs)

        if not df.empty:
            df = df.explode('highlight').reset_index(drop=True)
            df['processed_text'] = df['highlight'].apply(preprocess_text)
            df['search_term'] = term

            # Save processed text to CSV
            df.to_csv(processed_text_filename, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
            print(f"Saved processed text for term '{term}'.")

        # Write log for the term
        doc_count = len(df) if 'df' in locals() else 0
        write_log_for_term(term, params, runtime, doc_count, term_subfolder)

        return df

    return None



# ---------------------- BibJSON helpers (post-processing) ----------------------

BIBJSON_VERSION = "0.2"

def _is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

def _norm_str(x):
    if _is_nan(x):
        return None
    s = str(x).strip()
    return s if s else None

def _parse_year_fields(row):
    """Try pub_year/year first; fall back to YYYY from coverDate."""
    y = row.get("pub_year", None) or row.get("year", None)
    if not _is_nan(y):
        ys = str(y).strip()
        if ys.isdigit():
            return int(ys)
    cd = row.get("coverDate", None)
    if not _is_nan(cd):
        m = re.match(r"^(\d{4})", str(cd).strip())
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def _authors_to_list(authors_raw):
    """
    Accepts list[str], list[dict], dict, str, NaN.
    Returns list[{'name': '...'}].
    """
    if _is_nan(authors_raw):
        return []
    if isinstance(authors_raw, list):
        vals = authors_raw
    elif isinstance(authors_raw, dict):
        vals = [authors_raw]
    elif isinstance(authors_raw, str):
        vals = [p.strip() for p in re.split(r";|, and | and |\|", authors_raw) if p.strip()]
    else:
        vals = [str(authors_raw)]

    out = []
    for a in vals:
        if isinstance(a, dict):
            name = a.get("name") or " ".join([_norm_str(a.get("given")), _norm_str(a.get("family")) or ""]).strip()
            if name:
                out.append({"name": name})
        else:
            n = _norm_str(a)
            if n:
                out.append({"name": n})
    return out

def _row_to_bibjson_record(row):
    doi = _norm_str(row.get("doi")) or ""
    doc_id = _norm_str(row.get("doc_id")) or ""
    title = _norm_str(row.get("title")) or ""
    journal = _norm_str(row.get("journal")) or _norm_str(row.get("source"))
    publisher = _norm_str(row.get("publisher"))
    year = _parse_year_fields(row)
    url = _norm_str(row.get("url")) or _norm_str(row.get("landing_page"))
    rtype = row.get("type")
    if _is_nan(rtype):
        rtype = "article"

    rec = {
        "id": doi or doc_id or "",     # stable id preference: DOI > doc_id
        "type": rtype,
        "title": title,
        "author": _authors_to_list(row.get("authors") if "authors" in row else row.get("author")),
        "year": year,
        "journal": journal,
        "publisher": publisher,
        "identifier": [],
    }
    if doi:
        rec["identifier"].append({"type": "doi", "id": doi})
    if url:
        rec["link"] = [{"url": url, "content_type": "text/html"}]
    return rec

def build_bibjson_from_snippets_csv(
    csv_path,
    out_bibjson_path,
    out_map_csv=None,
    collection_name=None,
    chunksize=100_000,
    title_year_fallback=True,
):
    """
    Post-process an existing {term}_processed_text.csv (snippet-level) into:
      1) {term}.bibjson (one record per unique document)
      2) optional {term}_bib_map.csv mapping snippet rows -> bibjson_id (doi/doc_id)

    - Dedup key order: DOI -> doc_id -> (title, year) if enabled
    - Chunked reading to scale to very large files
    """
    if collection_name is None:
        collection_name = "xDD-collection"

    # Dedup trackers
    seen_doi = set()
    seen_docid = set()
    seen_title_year = set()

    # Accumulators
    bib_records = []
    map_rows = []  # [(global_row_index, bib_id)]

    global_row_idx = 0

    # Iterate in chunks to keep memory in check
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str, keep_default_na=True):
        # Normalize known fields
        # (All columns are strings because dtype=str; treat "nan" as NaN-equivalent)
        def _nanify(s):
            return None if (s is None or str(s).lower() == "nan" or str(s).strip() == "") else s

        # Compute doc keys per row
        dois = chunk.get("doi")
        docids = chunk.get("doc_id")
        titles = chunk.get("title")
        years = None

        # If we might need title-year fallback, precompute years from fields if missing
        if title_year_fallback and (titles is not None):
            # Build a minimal row-like object for year parsing
            # Use vectorized attempts where possible, then fallback row-wise
            if "pub_year" in chunk.columns and chunk["pub_year"].notna().any():
                years = pd.to_numeric(chunk["pub_year"], errors="coerce")
            elif "year" in chunk.columns and chunk["year"].notna().any():
                years = pd.to_numeric(chunk["year"], errors="coerce")
            else:
                # fallback: parse year from coverDate if present
                if "coverDate" in chunk.columns:
                    years = chunk["coverDate"].astype(str).str.extract(r"^(\d{4})")[0]
                    years = pd.to_numeric(years, errors="coerce")
                else:
                    years = pd.Series([math.nan] * len(chunk), index=chunk.index)

        for i, row in chunk.iterrows():
            # Build a dict-like row access for _row_to_bibjson_record
            row_dict = {col: _nanify(val) for col, val in row.items()}

            doi = _norm_str(row_dict.get("doi"))
            doc_id = _norm_str(row_dict.get("doc_id"))

            # Decide bib_id and whether it's new
            bib_id = None
            is_new = False

            if doi:
                bib_id = doi
                if bib_id not in seen_doi:
                    is_new = True
                    seen_doi.add(bib_id)
            elif doc_id:
                bib_id = doc_id
                if bib_id not in seen_docid:
                    is_new = True
                    seen_docid.add(bib_id)
            elif title_year_fallback:
                title = _norm_str(row_dict.get("title"))
                # use precomputed years where possible
                year_val = None
                if years is not None:
                    try:
                        yv = years.loc[i]
                        if not (isinstance(yv, float) and math.isnan(yv)):
                            year_val = int(yv)
                    except Exception:
                        year_val = None
                if title and year_val:
                    bib_id = f"{title}::{year_val}"
                    if bib_id not in seen_title_year:
                        is_new = True
                        seen_title_year.add(bib_id)

            # If we still don't have a bib_id, synthesize a stable one from doc_id or row index
            if not bib_id:
                bib_id = doc_id or f"row-{global_row_idx}"

            # Record mapping for this snippet row
            map_rows.append((global_row_idx, bib_id))

            # If it's a new document, build a BibJSON record
            if is_new:
                rec = _row_to_bibjson_record(row_dict)
                # Ensure the 'id' of the bib record matches our chosen bib_id (so map aligns)
                rec["id"] = bib_id
                bib_records.append(rec)

            global_row_idx += 1

    # Write BibJSON collection
    bibjson_payload = {
        "collection": collection_name,
        "metadata": {
            "created": datetime.utcnow().isoformat() + "Z",
            "generator": "xdd_api.py (post-processing)",
            "version": BIBJSON_VERSION
        },
        "records": bib_records
    }
    with open(out_bibjson_path, "w", encoding="utf-8") as f:
        json.dump(bibjson_payload, f, ensure_ascii=False, indent=2)

    # Optional: write snippet→bib map
    if out_map_csv:
        pd.DataFrame(map_rows, columns=["snippet_row", "bibjson_id"]).to_csv(out_map_csv, index=False)

    print(f"[BIBJSON] Wrote {len(bib_records)} records → {out_bibjson_path}")
    if out_map_csv:
        print(f"[BIBJSON] Wrote snippet→bib map → {out_map_csv}")

def build_bibliography_for_term(root_folder, term):
    term_folder = os.path.join(root_folder, term.lower())
    csv_path = os.path.join(term_folder, f"{term.lower()}_processed_text.csv")
    out_bib = os.path.join(term_folder, f"{term.lower()}.bibjson")
    out_map = os.path.join(term_folder, f"{term.lower()}_bib_map.csv")
    build_bibjson_from_snippets_csv(
        csv_path=csv_path,
        out_bibjson_path=out_bib,
        out_map_csv=out_map,
        collection_name=f"xDD-{term.lower()}",
        chunksize=100_000,          # tune if needed
        title_year_fallback=True    # keep on to capture non-DOI docs
    )
