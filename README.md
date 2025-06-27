# GeoNLP: Semantic Text Analysis for Geoscience Literature

A collection of Jupyter notebooks and Python tools for extracting, analyzing, and visualizing semantic relationships in geoscience literature using NLP, entropy-based metrics, and co-occurrence networks.

Developed as part of my MSc thesis in geology and NLP at the University of Saskatchewan.

---

## 🌐 Live Demo & Thesis

- 🔗 Web App: [GeoNLP Portal](https://geo-nlp-portal.onrender.com/)
- 📄 MSc Thesis: [View PDF](https://www.terra-datasystems.com/msc-thesis.pdf)
- 👨‍💻 Personal Site: [www.terra-datasystems.com](https://www.terra-datasystems.com)

---

## 📁 Project Structure

This repo is organized as follows:

```bash
project/
├── logs/
│
├── notebooks/
│   ├── 1_xdd_data_extraction.ipynb     # Pull text from xDD and format snippets
│   ├── 2_calculate_statistics.ipynb    # Compute counts, entropy, MI, co-occurrence
│   └── 3_import_to_postgres.ipynb      # Load stats into PostgreSQL
│
├── src/
│   ├── .env							# environment credentials - not shared, need to supplier your own
│   ├── database.py                     # SQLAlchemy DB connection and helpers
│   ├── logging_utils.py                # Logging utilities to record function processing results
│   ├── nlp_statistics.py               # Information-theoretic statistics calculations
│   ├── text_processing.py              # Text preprocessing functions
│   └── xdd_api.py		                # Main xDD API Snippet call functions and helper functions
│
├── environment.yml                     # Python dependencies
├── LICENSE
└── README.md
````

---

## 🔍 Notebooks Overview

### `1_xdd_data_extraction.ipynb`

* Connects to the [xDD API](https://xDD.wisc.edu/) to retrieve text snippets for geological search terms.
* Parses and saves results to local `.csv` files by term.

### `2_calculate_statistics.ipynb`

* Computes statistical relationships from the corpus:

  * Word and pair frequencies
  * Joint entropy, mutual information
  * Co-occurrence probabilities

### `3_import_to_postgres.ipynb`

* Loads all calculated statistics into a PostgreSQL database.
* Ensures tables for `term_stats`, `term_pairs`, and `term_cooccurrence` are created and filled.

---

## 📊 Visualizing the Results

After importing the data, use the web app to explore semantic relationships:

1. Run the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

2. Open in browser:
   [http://localhost:8000](http://localhost:8000)

3. Search for a geoscience term like `"volcanic arc"`.

4. Use the interactive tools to:

   * Filter edges by co-occurrence probability
   * Highlight nodes by category
   * Export visualizations or download the underlying data

---

## 💻 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/geonlp-project.git
cd geonlp-project
```

### 2. Set Up the Conda Environment

```bash
conda create -n geonlp_env python=3.11
conda activate geonlp_env
pip install -r requirements.txt
```

> Note: You may need to install PostgreSQL and `psycopg2`:

```bash
conda install -c conda-forge psycopg2
```

### 3. Set Up Your `.env` File

Create a `.env` file in the root directory with:

```
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=geonlp
```

---

## 🧠 Built With

* 🐍 **Python** (pandas, SQLAlchemy, requests, numpy)
* 📚 **Jupyter Notebooks**
* 🗃️ **PostgreSQL** for structured data storage
* 🌐 **FastAPI** for serving the visualization app
* 📈 **Pyvis** for interactive semantic networks
* 📊 **Entropy & MI** as information-theoretic metrics

---

## 📘 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 👋 Acknowledgments

* Developed as part of my MSc at the University of Saskatchewan.
* Supervised by \Bruce Eglington]
* Based on data from the [xDD Project](https://geodeepdive.org/)

---

## 🙋‍♀️ Questions? Contributions?

If you'd like to contribute or have feedback, feel free to:

* Open an issue
* Submit a pull request
* Contact me at [dheasman@smcg-services.com](mailto:dheasman@smcg-services.com)
