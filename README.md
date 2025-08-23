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

Thanks — that's a great distinction, and it's worth reflecting in the tone and purpose of your README.

Here's a **refined replacement section** for your `README.md`:

---

## 🌐 Explore the Results Online

After processing and importing the data, you can explore the semantic relationships using the **web-based GeoNLP App** — no coding required.

This app was built to **democratize access** to geoscientific text mining tools, making them available to a broader audience regardless of technical background.

### 🔗 Web App Access

You can explore the interactive co-occurrence networks and mutual information results directly via:

👉 **[GeoNLP App](https://geo-nlp-portal.onrender.com/)**

### 🧭 Key Features:

* Search for geological terms (e.g., `"volcanic arc"`)
* Visualize semantic networks of related words
* Filter edges by co-occurrence probability
* Highlight nodes by type (e.g., mineral, lithology, structure)
* Download raw statistics for external use

> 🧪 *This web app complements the code in this repository, offering a no-code interface for geoscientists, students, and the public.*

---

## 💻 Installation - (Unneccessary if using the App)

### 1. Clone the Repo

```bash
git clone https://github.com/DHeasmanGDS/msc-geonlp-notebooks
cd msc-geonlp-notebooks
```

Great — since you're using a `.yml` file to manage your environment (which is the better approach for reproducibility), here’s the updated version of that section for your `README.md`:

---

### 2. Set Up the Conda Environment

Create the environment from the included `.yml` file:

```bash
conda env create -f environment.yml
conda activate msc-geonlp-notebooks
```

> 📝 **Note**: If the environment file doesn't install everything perfectly (e.g., issues with `psycopg2`), you can manually install missing packages:

```bash
conda install -c conda-forge psycopg2
```

---

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

## ⚖️ Licensing & Attribution

**Code, methods, and documentation in this repository**
- Licensed under the **MIT License** (see `LICENSE`). You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell the software, subject to the MIT terms.

**xDD data and anything derived directly from xDD content**
- Licensed under **CC BY-NC 4.0** **by xDD**. This applies to all xDD outputs you obtain via the API or platform (e.g., snippets, metadata records, and any files that reproduce or closely track those outputs such as term-level snippet CSVs, BibJSON built from xDD metadata, example figures based on xDD results, etc.). Commercial use is **not** permitted under CC BY-NC. Cite xDD when you use these materials.  
  > xDD states: “All xDD output is licensed under CC-BY-NC.” :contentReference[oaicite:0]{index=0}

**Attribution example**
> Data and text snippets from xDD (University of Wisconsin–Madison), CC BY-NC 4.0.  
> Peters SE, Ross IA, Rekatsinas ML (2023), “xDD: a platform for text and data mining from scholarly publications.”

**Notes**
- Some documents indexed by xDD originate from third-party publishers. When redistributing any content, ensure compliance with the **publisher’s** terms in addition to xDD’s license.  
- This statement is informational and not legal advice.


---

## 👋 Acknowledgments

* Developed as part of my MSc at the University of Saskatchewan.
* Supervised by Bruce Eglington
* Based on data from the [xDD](https://geodeepdive.org/)

* Peters SE, Ross IA, Rekatsinas ML (2023) xDD: a platform for text and data mining from scholarly publications.

---

## 🙋‍♀️ Questions? Contributions?

If you'd like to contribute or have feedback, feel free to:

* Open an issue
* Submit a pull request
* Contact me at [dheasman@smcg-services.com](mailto:dheasman@smcg-services.com)
