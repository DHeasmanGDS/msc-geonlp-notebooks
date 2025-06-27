################################################################################
# Filename: logging_utils.py
#
# Purpose: 
# This script contains functions designed to log the results and errors 
# from other scripts, ensuring traceability of API calls and data processing.
#
# Author: Drew Heasman, P.Geo
# Last Updated: 26-06-2025
#
# Organization: University of Saskatchewan
################################################################################

################################################################################
# Future Improvements
################################################################################

# - Implement logging levels (INFO, WARNING, ERROR) for better debugging.
# - Use Python’s built-in `logging` module instead of manual file writing.
# - Support logging to a central database or cloud service.
# - Add log rotation to prevent excessive file growth.
# - Format logs in JSON for easier parsing and analysis.

################################################################################
# Libraries
################################################################################

# Standard Libraries
import os
from datetime import datetime

################################################################################
# Global Variables
################################################################################

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

################################################################################
# Functions
################################################################################

def write_log_for_term(term, params, runtime, doc_count, output_folder):
    """
    Logs details of an API call for a specific search term.

    This function creates or appends to a log file, storing:
    - The date and time of execution.
    - The search term used.
    - API parameters applied.
    - The number of retrieved documents.
    - The runtime duration.

    Parameters:
    -----------
    term : str
        The search term associated with the API request.

    params : dict
        Dictionary of API parameters used in the request.

    runtime : float
        The execution time of the API request in seconds.

    doc_count : int
        The number of documents retrieved in the request.

    output_folder : str
        The directory where the log file should be saved.

    Returns:
    --------
    None
        The function writes to a log file but does not return any values.

    Example Usage:
    --------------
    >>> params = {"query": "volcanic arc", "limit": 100, "date_range": "2000-2023"}
    >>> write_log_for_term("volcanic arc", params, 2.54, 97, "logs/")
    
    Features:
    ---------
    - **Timestamp Logging:** Uses a consistent timestamp format to track execution times.
    - **Appending Log Entries:** Ensures that previous logs are preserved and new runs are added.
    - **Readable Log Format:** Entries are structured clearly for easy debugging and tracking.

    Notes:
    ------
    - The log file is named `{search_term}_log.txt` and is stored in `output_folder`.
    - The timestamp format is defined globally as `TIMESTAMP_FORMAT`.
    - The log file is encoded in UTF-8 to ensure compatibility with special characters.
    """
    
    log_filepath = os.path.join(output_folder, f"{term}_log.txt")
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

    with open(log_filepath, "a", encoding="utf-8") as log_file:
        log_file.write(f"==== Latest Run ====\n")
        log_file.write(f"Date: {timestamp}\n")
        log_file.write(f"Search Term: {term}\n")
        log_file.write(f"API Parameters Used: {params}\n")
        log_file.write(f"Documents Retrieved: {doc_count}\n")
        log_file.write(f"Runtime (seconds): {runtime:.2f}\n")
        log_file.write("====================\n\n")

def write_statistics_log(term, total_snippets, total_documents, total_tokens, unique_terms, unique_publishers, runtime, log_folder="logs"):
    """
    Logs the corpus summary statistics for co-occurrence processing.

    Parameters are passed directly from your calculate_corpus_statistics().
    """
    ensure_log_folder(log_folder)        

    log_filepath = os.path.join(log_folder, "statistics_log.txt")
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

    with open(log_filepath, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] Term: {term}\n")
        log_file.write(f"  Total snippets: {total_snippets}\n")
        log_file.write(f"  Total documents: {total_documents}\n")
        log_file.write(f"  Total tokens: {total_tokens}\n")
        log_file.write(f"  Unique terms: {unique_terms}\n")
        log_file.write(f"  Unique publishers: {unique_publishers}\n")
        log_file.write(f"  Runtime (s): {runtime:.2f}\n")
        log_file.write("---------------------------------------------------\n")


def write_general_log(message, log_filename="general_log.txt", log_folder="logs"):
    """
    Writes a general log entry for tracking events, errors, or system updates.

    This function appends messages to a log file, recording:
    - The date and time of execution.
    - A user-defined log message.

    Parameters:
    -----------
    message : str
        The message to be logged.

    log_filename : str, optional
        The name of the log file where the message will be saved (default: `"general_log.txt"`).

    log_folder : str, optional
        The directory where the log file should be stored (default: `"logs"`).

    Returns:
    --------
    None
        The function writes to a log file but does not return any values.

    Example Usage:
    --------------
    >>> write_general_log("Database connection successful.")
    >>> write_general_log("Processed 500 documents.", log_filename="processing_log.txt")

    Features:
    ---------
    - **Timestamped Logging:** Each log entry includes a timestamp for tracking when the event occurred.
    - **Automatic Directory Creation:** Ensures the specified log folder exists before writing.
    - **Flexible Logging:** Allows specifying different log filenames and folders if needed.
    - **UTF-8 Encoding:** Ensures compatibility with international characters and symbols.

    Notes:
    ------
    - The timestamp format is defined globally as `TIMESTAMP_FORMAT`.
    - If the `log_folder` does not exist, it is automatically created.
    - Each log entry is appended to avoid overwriting previous logs.
    """
    
    # Ensure the log directory exists
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    log_filepath = os.path.join(log_folder, log_filename)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

    with open(log_filepath, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def write_error_log(error_message, error_type="ERROR", log_folder="logs"):
    """
    Logs errors, warnings, or informational messages to a dedicated error log file.

    This function appends error-related messages to a log file, recording:
    - The date and time of execution.
    - The type of error (e.g., INFO, WARNING, ERROR).
    - The actual error message.

    Parameters:
    -----------
    error_message : str
        The message describing the error or warning.

    error_type : str, optional
        The severity level of the error (default: `"ERROR"`).
        - `"INFO"` : Informational message.
        - `"WARNING"` : A non-critical issue that may need attention.
        - `"ERROR"` : A serious issue that requires immediate attention.

    log_folder : str, optional
        The directory where the log file should be stored (default: `"logs"`).

    Returns:
    --------
    None
        The function writes to a log file but does not return any values.

    Example Usage:
    --------------
    >>> write_error_log("Failed to connect to the database.", error_type="ERROR")
    >>> write_error_log("Memory usage is high.", error_type="WARNING")
    >>> write_error_log("Script executed successfully.", error_type="INFO")

    Features:
    ---------
    - **Timestamped Logging:** Each log entry includes a timestamp for tracking when the issue occurred.
    - **Automatic Directory Creation:** Ensures the specified log folder exists before writing.
    - **Flexible Logging Levels:** Supports `"INFO"`, `"WARNING"`, and `"ERROR"` types for better categorization.
    - **UTF-8 Encoding:** Ensures compatibility with international characters and symbols.
    - **Persistent Logging:** Appends new entries to avoid overwriting previous logs.

    Notes:
    ------
    - The timestamp format is defined globally as `TIMESTAMP_FORMAT`.
    - If the `log_folder` does not exist, it is automatically created.
    - Log messages are stored in `"error_log.txt"` inside the specified log folder.
    """
    
    error_filename = "error_log.txt"

    # Ensure the log directory exists
    ensure_log_folder(log_folder)

    log_filepath = os.path.join(log_folder, error_filename)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

    with open(log_filepath, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] [{error_type}] {error_message}\n")


def ensure_log_folder(log_folder):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
