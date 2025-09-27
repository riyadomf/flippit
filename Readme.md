<!-- ```uvicorn main:app --reload```
```npm start``` -->

---

## Flippit: A Hybrid ML & LLM Real Estate Scoring Engine

### **1. Exploratory Data Analysis (EDA): Uncovering Market Signals**

A comprehensive Exploratory Data Analysis was performed to understand the underlying patterns of the Warren, MI real estate market. The goal was to identify key value drivers and potential data quality issues before modeling. The full analysis can be viewed in the [Google Colab Notebook](https://colab.research.google.com/drive/1KgiYSGMDJZ3ZAEdd5Egh1Up6zoNJ3zQR?usp=sharing).

**Key Findings:**

*   **Data Quality & Integrity:**
    *   **Missing Values:** A substantial number of null values were identified in key predictive columns like `text`, `beds`, `tax`, and `hoa_fee` in the `sold_properties` dataset.
    *   **Duplicates:** The `for_sale_properties` dataset contained duplicate `property_id` entries, which were resolved by keeping only the most recent listing for each property.
    *   **Outliers:** Extreme and unrealistic outliers were found in `price_per_sqft` (e.g., values below $20 or above $400). These were identified as likely data errors (e.g., family sales, teardown properties) and filtered out to prevent them from corrupting the model.

*   **Market Structure & Value Drivers:**
    *   **Price Distribution:** The distribution of `sold_price` was found to be bimodal, with distinct peaks around the **$150k** and **$250k** price points, suggesting two primary market segments.
    *   **Location is Paramount:** A clear tiered hierarchy exists for `price_per_sqft` across different neighborhoods, with values systematically increasing from **Southeast Warren < Southwest Warren < Northwest Warren < Northeast Warren**. This confirms that location is a dominant feature.
    *   **The Law of Diminishing Returns:** Analysis of `price_per_sqft` by `size_range` revealed a crucial non-linear relationship: smaller homes (<1000 sqft) consistently have the highest median price per square foot, and this value decreases as home size increases. This insight proves that a simple linear model would be insufficient.
    *   **Textual Signals:** Preliminary NLP analysis showed a strong correlation between keywords in the `text` description and final sale price. Properties described with terms like "granite" or "renovated" commanded a significant price premium, validating the use of text features.
    *   **External Benchmark:** A strong linear relationship between `realtor.com`'s `estimated_value` and the actual `sold_price` confirmed its value as a reliable predictive feature for the model.

---

### **2. Data Preprocessing**

Based on the EDA findings, I created a multi-stage preprocessing pipeline ensuring that all data, whether for training or live inference, is handled consistently to prevent training-serving skew.

**The pipeline follows a strict order of operations:**

1.  **Filtering and Deduplication:**
    *   Duplicate `property_id` entries are removed, keeping only the most recent listing based on `list_date`.

2.  **Outlier Removal (Training Data Only):**
    *   To create a clean training signal, properties with unrealistic `price_per_sqft` values (outside a defined range like $20-$350) are removed. This crucial step is only applied to the training dataset to avoid accidentally filtering out valid properties during live scoring.
        [Example outlier](https://www.realtor.com/realestateandhomes-detail/26304-Patricia-Ave_Warren_MI_48091_M99820-31519
        )
    *   Rows with an excessive number of missing values in key columns (e.g., `list_price`, `estimated_value`, and `year_built` both null) are dropped from the training set to improve model stability.

3.  **Intelligent Imputation:**
    *   A **grouped imputation** strategy was implemented. Missing numerical and categorical values are filled using the median or mode, respectively, grouped by both `zip_code` and `size_range`.
    *   A **global fallback** (e.g., city-wide median) is learned from the training data to handle cases where a new, unseen `zip_code` / `size_range` combination appears during inference.

---

### **3. Feature Engineering**

The core of the model's predictive power comes from creating new, more informative features from the raw data.

1.  **Basic Feature Creation:**
    *   **`property_age`**: Calculated as `current_year - year_built` to create a simple, powerful numerical feature for the model.
    *   **`total_baths`**: A combined feature (`full_baths + 0.5 * half_baths`) that provides a better signal of a property's utility than two separate columns.
    *   **`size_range`**: A categorical feature (`<1000`, `1000-1499`, etc.) derived from `sqft` to allow the model to learn the non-linear value of space discovered in the EDA.

2.  **Advanced Feature Creation (Hybrid ML + LLM):**
    *   **LLM-Powered Features:** An offline script uses an LLM to analyze the raw `text` description of each property, generating a structured set of features:
        *   `llm_quality_score`: A 1-10 rating of the property's finishes and appeal. **This is the primary new feature for the ARV model.**
        *   `renovation_level`: A categorical assessment of the work needed.
        *   `llm_risk_score`: A 1-10 rating of investment risk based on textual red flags.
    *   **Interaction Features:** To help the model learn more complex patterns, interaction features were created:
        *   `sqft_x_quality_score`: This crucial feature allows the model to understand that a high `llm_quality_score` is worth more on a large house than a small one.

3.  **Final Processing:**
    *   **One-Hot Encoding:** All final categorical features (`zip_code`, `neighborhoods`, `size_range`, `renovation_level`) are one-hot encoded to be used in the Gradient Boosting model.
    *   **Dimensionality Reduction:** Redundant or noisy raw features (e.g., `text`, `half_baths`) are dropped, as their predictive signal has been captured in the new engineered features. The final feature set is curated based on the `BASE_MODEL_FEATURES` config to ensure a lean and powerful model.




### **4. Methodology: A Hybrid Approach for Scoring**

The core of the Flippit service is a scoring engine that calculates the potential Return on Investment (ROI) for each for-sale property. The fundamental formula is:

**Expected Profit = `Estimated Resale Price` - (`List Price` + `Renovation Cost` + `Other Cost`)**
**ROI = (`Expected Profit` / `Total Cash Invested`) * 100**

To solve this, I broke down the problem into distinct, modular components, each of which was improved iteratively.

#### **V1: Heuristic Baseline Model**

The initial approach was a rule-based system (MVP).
*   **Resale Price Estimation:** Calculated using a simple `price_per_sqft` metric derived from the `sold_properties` data, grouped by ZIP code and property size range.
*   **Renovation Cost:** Estimated using a tiered heuristic based on the property's age (`year_built`) and the presence of simple keywords (e.g., "fixer-upper" vs. "updated") in the text description.
*   **Other Cost:** Selling Cost (Percentage to agent or others) + Carrying Cost
*   **Risk & Grading:** A simple risk score was calculated based on factors like `days_on_mls` and `year_built`. An A-F grade was then assigned using fixed ROI and risk thresholds.

**Limitation:** This approach was rigid and could not capture the complex, non-linear relationships in the market. The keyword analysis was basic and missed significant nuance in the property descriptions.

#### **V2: Machine Learning for Price Prediction**

Then I replaced the heuristic resale price calculation with a predictive machine learning model.

*   **Model Selection: Gradient Boosting (XGBoost)**
    The XGBoost model was deliberately chosen over simpler and more complex alternatives for several key reasons:
    1.  **vs. Linear/Polynomial Regression:** Real estate value is not linear. Our EDA proved that the value of an extra square foot diminishes as a house gets larger. Linear models cannot capture this nuance, whereas a tree-based model like XGBoost excels at modeling complex, non-linear relationships and feature interactions (e.g., the value of an extra bathroom is higher in a 4-bedroom house than a 2-bedroom one).
    2.  **vs. Neural Networks:** For the size and structure of our dataset (tabular CSV data), Gradient Boosting models consistently outperform Neural Networks. Neural Networks typically require much larger datasets to be effective and are more difficult to interpret.

*   **Training & Validation Strategy:**
    The model's goal is to predict the **After-Repair Value (ARV)**. Therefore, simply training on all sold data is insufficient. A specialized validation strategy was employed:
    1.  The model was **trained on the entire training dataset** to learn broad market patterns.
    2.  It was **evaluated against a "Gold Standard" subset** of the holdout test data. This subset included only properties that were already in excellent, renovated condition (`renovation_level='Cosmetic'`, `llm_quality_score >= 6`).
    This ensures our key performance metric (Mean Absolute Error) accurately reflects the model's ability to predict the final price of a finished, desirable product.

#### **V3: The Hybrid Approach - Augmenting ML with LLM Intelligence (Latest Approach)**

The latest and most powerful version of the scoring engine is a hybrid system that combines the predictive power of XGBoost with the nuanced text understanding of a Large Language Model (LLM).

**The Core Insight:** A simple keyword-based system is brittle. An LLM, by contrast, understands the semantic difference between "needs a new kitchen faucet" (minor) and "needs a full kitchen remodel" (major). We leveraged this by using an LLM as an advanced feature engineering engine.

1.  **Offline LLM Feature Generation:**
    An offline script processes both the `sold_properties.csv` and `for_sale_properties.csv` datasets. Using a carefully crafted few-shot prompt, the LLM analyzes the `text` description of each property and outputs a structured JSON object containing:
    *   `renovation_level`: A categorical assessment ('Cosmetic', 'Medium', 'Heavy', 'Gut').
    *   `quality_score`: A 1-10 rating of the property's finishes and appeal.
    *   `risk_score`: A 1-10 rating of textual red flags (e.g., "as-is", "structural issues").
    *   `positive_features` & `negative_features`: Lists of keywords for the final explanation.
    This step creates new, enriched datasets that are then used for training and scoring.

2.  **Resale Price Estimation (Hybrid):**
    *   The `llm_quality_score`, `sqft_x_quality_score`, and `renovation_level` are now core features used to train the XGBoost ARV model.
    *   At inference time, we ask the model a precise question: "What is the predicted price of this property if we **manually set its `llm_quality_score` to 9 (high-end) and its `renovation_level` to 'Cosmetic'**?" This simulation forces the model to predict the true ARV, resulting in a much more accurate and realistic resale price.

3.  **Renovation Cost Estimation (LLM-Driven):**
    *   The heuristic keyword model was replaced entirely. The `renovation_level` provided by the LLM is now mapped to a pre-defined cost-per-square-foot (`RENOVATION_COST_MAP`), resulting in a far more nuanced and context-aware cost estimate.

4.  **Risk Assessment & Grading (Hybrid):**
    *   The `assess_risk` function now uses the `llm_risk_score` as its baseline, augmenting it with structured data points the LLM can't see (`days_on_mls`, `year_built`, etc.).
    *   The final `assign_grade` function uses a risk-averse model that requires both a high ROI and a low final risk score to achieve a top grade, ensuring that only the most promising and safest deals are recommended to the user.


