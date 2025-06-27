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

# Ensure the 'src' directory is included in the import path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

# Internal Module Imports
from text_processing import preprocess_text
from logging_utils import write_log_for_term

################################################################################
# Global Variables
################################################################################

# XDD API url endpoint
API = "https://xdd.wisc.edu/api/v2/snippets"

################################################################################
# Main Function Call
################################################################################

def process_and_save_results(search_terms, params, headers, root_folder, overwrite=False):
    """
    Sequentially processes search terms, saves results, and provides periodic status updates.

    Parameters:
    -----------
    search_terms : list
        A list of search terms to query via the xDD API.
    params : dict
        API request parameters.
    headers : dict
        API request headers, including authentication.
    root_folder : str
        Directory to save processed results.
    overwrite : bool, optional
        Whether to overwrite existing processed files (default: False).

    Returns:
    --------
    list of pd.DataFrame:
        DataFrames for each successfully processed term.
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


