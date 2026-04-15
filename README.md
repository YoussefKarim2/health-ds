## Business Questions

**Q1 (Business – Resource Allocation):** How can hospital administrators accurately predict which diabetic patients are at highest risk of 30-day readmission to target them with expensive post-discharge support (e.g., home visits) and avoid regulatory penalties?

**Q2 (Research - Clinical Impact):** Does a documented "change in medication" (e.g., insulin dosage adjustment) during the primary admission correlate with a statistically significant increase or decrease in readmission risk when controlling for patient severity?

## Project Overview

The project work is undertaken individually, allowing students to showcase a profound understanding of data science principles and the application of the CRISP-DM lifecycle in building scalable software solutions.

The project framework comprises the following core components:

1. **Reproducible Code Repository (GitHub)**
   - **Version Control:** You must initialize a private GitHub repository at the start of your project. Regular commits should demonstrate the evolution of your work (not just a single upload at the deadline).
   - **Scalable Structure:** You are required to move beyond a "single massive notebook." Your repository must follow a standard Data Science project structure (e.g., Cookiecutter Data Science):
     - `data/`: Storing raw vs. processed datasets.
     - `notebooks/`: For Exploratory Data Analysis (EDA) and visualization experiments.
     - `src/`: For reusable Python scripts. You must refactor your core Data Cleaning and Feature Engineering functions from your notebook into .py scripts (e.g., `src/processing.py`) and import them back into your notebook.

2. **Jupyter Notebook Implementation:**
   - **Narrative Flow:** Develop a "Final Pipeline" notebook that imports your functions from `src/` and executes the full project lifecycle sequentially.
   - **Data Scraping & Augmentation:** You are required to implement Web Scraping techniques (using BeautifulSoup, Selenium, or Scrapy) to collect supplementary data.
     - **Goal:** This scraped data must be merged with your primary dataset to increase its size or add new features (e.g., scraping "Average Temperature" by city to merge with a "Sales" dataset).
     - **Performance:** Compare the performance of sequential vs. parallel scraping if applicable.
   - **Data Audit:** Perform a rigorous Data Quality Assessment on the combined dataset (checking for missingness, outliers, and schema constraints) before processing.
   - **Exploratory Analysis:** Visualize distributions and correlations to justify your feature selection choices.
   - **Modeling Pipeline:** Implement the full modeling workflow (Train/Test Split, Cross-Validation, Hyperparameter Tuning) using `sklearn.pipeline`.

3. **Formal Project Report (1500 Words):**

   Write a professional report detailing the application of the CRISP-DM lifecycle to your problem. The report must include:

   - **Business Understanding:** Clearly state the business/research questions and the business value (Phase 1).
   - **Big Data & Scalability Analysis (Critical):** Critically evaluate the practical challenges of working with Big Data.
     - Discuss the selection and application of appropriate tools (e.g., justifying why you used Pandas vs. Spark, or how your pipeline would need to change if the dataset grew to 10TB).
     - Analyze the computational complexity of your chosen algorithms.
   - **Data Methodology:**
     - Provide a comprehensive description of the created dataset, including its parameters.
     - Detail the Data Scraping techniques employed (source, tools used, challenges faced).
     - Describe the Merging Strategy used to combine the scraped data with the primary dataset.
     - Describe your data audit findings and the specific cleaning steps applied (Phase 2 & 3).
     - Utilize suitable visualization techniques, accompanied by supporting plots.
   - **Modeling Strategy:** Justify your choice of algorithms and evaluation metrics (Phase 4 & 5).
   - **Ethical Considerations:** A dedicated section analyzing potential bias in your dataset or model (referencing the Data Ethics Framework).
   - **Conclusion:** Summarize the evidence generated and how it answers the business/research questions.

   **Report Structure Requirements:**
   - Cover Page (Project Title, Student Name/ID, TKH & Coventry Logos, Submission Date).
   - Abstract, Table of Contents, Table of Figures.
   - Introduction, Methodology (including Scraping), Results, Discussion, Ethics, Conclusion.

## Submission Deliverables

By the deadline specified on Moodle, each student is required to submit:

1. GitHub Repository Link: (Access must be granted to instructors).
2. Jupyter Notebook: (The final run-through).
3. Project Report: (PDF format).
4. Presentation: (10-minute slide deck).
5. A3 Poster: (Created via Canva/Figma).