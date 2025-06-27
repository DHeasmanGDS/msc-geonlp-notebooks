################################################################################
# Filename: text_processing.py
#
# Purpose: 
# This script contains functions designed to preprocess text for various 
# text mining and NLP applications, including tokenization, stopword removal, 
# lemmatization, and special character cleaning.
#
# Author: Drew Heasman, P.Geo
# Last Updated: 26-06-2025
#
# Organization: University of Saskatchewan
################################################################################

################################################################################
# Future Improvements
################################################################################

################################################################################
# Libraries
################################################################################

# Standard Libraries
import re
import string
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

################################################################################
# Global Variables
################################################################################

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Define a standard stopwords set
stop_words = set(stopwords.words('english'))
additional_stop_words = {'fig', 'et', 'al', 'eg'}
stop_words.update(additional_stop_words)

################################################################################
# Main Function Call
################################################################################

def preprocess_text(text, stop_words=stop_words):
    """
    Cleans and normalizes a given text string by applying multiple preprocessing steps.

    This function performs the following text-processing steps:
    - **Removes illegal characters and accents** using `remove_illegal_chars()`
    - **Converts text to lowercase** for uniformity.
    - **Removes numbers, punctuation, and special characters** for cleaner tokenization.
    - **Removes stopwords** to filter out common, non-informative words.
    - **Filters out single-character words** to remove noise.
    - **Removes extra spaces** to standardize formatting.

    Parameters:
    -----------
    text : str
        The input text string that needs to be preprocessed.

    stop_words : set, optional
        A set of stopwords to be removed from the text.
        Defaults to the standard stopwords set.

    Returns:
    --------
    str
        The cleaned and preprocessed text.

    Example Usage:
    --------------
    >>> preprocess_text("The quick brown fox jumps over 123 lazy dogs!")
    'quick brown fox jumps lazy dogs'

    >>> preprocess_text("Hello, this is an NLP test. Let's remove stopwords!", stop_words={"this", "is", "an"})
    'hello nlp test let remove stopwords'

    Features:
    ---------
    - **Custom Stopword Removal:** Uses a user-defined stopword set or a default one.
    - **Regex-Based Cleaning:** Removes unwanted characters efficiently.
    - **Handles Accents & Illegal Characters:** Uses `remove_illegal_chars()` for text normalization.
    - **Efficient String Processing:** Optimized for large text inputs.

    Notes:
    ------
    - This function assumes `remove_illegal_chars()` is defined elsewhere in the code.
    - Stopwords should be provided as a set for faster lookup and removal.
    - Single-character words (e.g., "a", "I") are removed unless they are meaningful terms.
    """
    
    text = remove_illegal_chars(text)  # Normalize and clean text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    
    # Remove stopwords
    if stop_words:
        text = ' '.join(word for word in text.split() if word not in stop_words)

    # Remove single-character words
    text = ' '.join(word for word in text.split() if len(word) > 1)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

################################################################################
# Functions
################################################################################

def remove_illegal_chars(text):
    """
    Cleans text by removing illegal characters, accents, and unwanted symbols.

    This function performs the following cleaning steps:
    - **Normalizes Unicode characters** to ensure consistency.
    - **Removes accents and diacritics** from letters (e.g., "café" → "cafe").
    - **Eliminates non-ASCII characters** to avoid encoding issues.
    - **Removes unwanted symbols** such as mathematical signs, degree symbols, etc.
    - **Reduces multiple spaces** to a single space for better readability.

    Parameters:
    -----------
    text : str
        The input text that needs to be cleaned.

    Returns:
    --------
    str
        The cleaned text with illegal characters removed.

    Example Usage:
    --------------
    >>> remove_illegal_chars("Café Déjà vu! ∑πΩ  42° ")
    'Cafe Deja vu 42 '

    >>> remove_illegal_chars("This text—contains (strange) symbols!@#$%^&*")
    'This text contains strange symbols'

    Features:
    ---------
    - **Handles Accents and Diacritics:** Converts characters with accents to their ASCII equivalents.
    - **Removes Non-Printable & Special Characters:** Strips out symbols that don't belong in standard text.
    - **Prevents Excess Spaces:** Ensures output text is neatly formatted with single spaces.
    - **Retains Basic Punctuation & Common Formatting:** Keeps periods, commas, and basic symbols intact.

    Notes:
    ------
    - Uses `unicodedata.normalize('NFKD', text)` to standardize Unicode representations.
    - Works well for text preprocessing in NLP applications, database inputs, and cleaning noisy text data.
    """
    
    text = unicodedata.normalize('NFKD', text)  # Normalize Unicode characters
    text = ''.join(c for c in text if not unicodedata.combining(c))  # Remove accents
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters

    # Remove specific unwanted characters (e.g., symbols)
    illegal_chars = re.compile(r'[^\w\s.,;\'\"!?()\-]')
    text = illegal_chars.sub('', text)

    # Reduce multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_text(text, unwanted_words=None, use_stemming=True):
    """
    Cleans text by removing unwanted words, handling plurals, and applying either lemmatization or stemming.

    This function performs the following text-processing steps:
    - **Tokenizes** the text into individual words.
    - **Converts text to lowercase** for uniformity.
    - **Removes unwanted words** (e.g., "httpdoc", "example") if provided.
    - **Applies stemming** (reducing words to root form) if `use_stemming=True`.
    - **Applies lemmatization** (converting words to dictionary form) if `use_stemming=False`.

    Parameters:
    -----------
    text : str
        The input text string to be cleaned.

    unwanted_words : list or set, optional
        A set or list of words to remove from the text.
        Defaults to `{"httpdoc", "example"}` if not provided.

    use_stemming : bool, optional
        - If `True`, applies **stemming** using `PorterStemmer()`.
        - If `False`, applies **lemmatization** using `WordNetLemmatizer()`.
        Defaults to `True`.

    Returns:
    --------
    str
        The cleaned and processed text.

    Example Usage:
    --------------
    >>> clean_text("The httpdoc example shows various tokens.", unwanted_words={"httpdoc", "example"})
    'show variou token'  # (With stemming)

    >>> clean_text("Running horses love playing.", use_stemming=False)
    'run horse love play'  # (With lemmatization)

    Features:
    ---------
    - **Custom Word Removal:** Users can define a list of unwanted words.
    - **Choice Between Stemming & Lemmatization:** More flexibility in text preprocessing.
    - **Efficient Tokenization & Filtering:** Uses `word_tokenize()` for precise text splitting.

    Notes:
    ------
    - `word_tokenize()` requires `nltk.download('punkt')`.
    - `get_wordnet_pos()` should be defined to improve lemmatization accuracy.
    - Stemming produces shorter root words, while lemmatization keeps dictionary-based forms.
    """
    
    if unwanted_words is None:
        unwanted_words = {"httpdoc", "example"}  # Default list of words to remove

    tokens = word_tokenize(text.lower())  # Tokenize text
    tokens = [word for word in tokens if word not in unwanted_words]  # Remove unwanted words

    # Apply either stemming or lemmatization
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    else:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]

    return ' '.join(tokens)

def get_wordnet_pos(word):
    """
    Maps POS (Part-of-Speech) tags to the format required by the WordNet lemmatizer.

    This function determines the POS tag of a word using `nltk.pos_tag()` and maps it to a 
    corresponding WordNet POS tag for more accurate lemmatization.

    Parameters:
    -----------
    word : str
        The word for which the POS tag needs to be determined.

    Returns:
    --------
    str
        The corresponding WordNet POS tag:
        - `"a"` for adjectives (ADJ)
        - `"n"` for nouns (NOUN)
        - `"v"` for verbs (VERB)
        - `"r"` for adverbs (ADV)
        - Defaults to `"n"` (noun) if the tag is not found.

    Example Usage:
    --------------
    >>> get_wordnet_pos("running")
    'v'  # Verb

    >>> get_wordnet_pos("beautiful")
    'a'  # Adjective

    >>> get_wordnet_pos("dog")
    'n'  # Noun

    Features:
    ---------
    - **Uses NLTK's POS Tagger**: `nltk.pos_tag()` provides context-aware tagging.
    - **Enhances Lemmatization Accuracy**: Helps WordNet lemmatizer return correct base forms.
    - **Efficient Mapping**: Converts POS tags to WordNet-compatible format using a dictionary lookup.

    Notes:
    ------
    - Requires `nltk.download('averaged_perceptron_tagger')` for POS tagging.
    - If the POS tag is not found in the mapping, it defaults to `"n"` (noun).
    """
    
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
