
import json
import os

def process_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Dictionary of Pre-computation explanations (The "Why")
    pre_explanations = {
        1: "### Step 1: Install Core Libraries\n**Why:** We are installing `pandas` and `numpy`. `pandas` is the standard library for data manipulation (DataFrames) in Python, and `numpy` provides support for efficient numerical operations and arrays. These are the building blocks of our analysis.",
        2: "### Step 2: Install Machine Learning Library\n**Why:** We install `scikit-learn`. This library provides the machine learning algorithms (Logistic Regression), preprocessing tools (scalers, encoders), and metrics (accuracy, ROC-AUC) we will use to build and evaluate our churn models.",
        3: "### Step 3: Install Excel Support\n**Why:** Our dataset is in the `.xlsx` (Excel) format. Pandas requires the `openpyxl` engine to read these files.",
        4: "### Step 4: Import Excel Engine\n**Why:** We explicitly import `openpyxl` to ensure the environment has access to it before we attempt to load the data.",
        5: "### Step 5: Import All Required Libraries\n**Why:** We import the necessary modules upfront. This includes:\n- **Data Handling:** `pandas`, `numpy`\n- **Modeling:** `LogisticRegression`, `Pipeline`\n- **Preprocessing:** `OneHotEncoder` (for text categories), `StandardScaler` (for numbers)\n- **Metrics:** `roc_auc_score`, `classification_report`\nThis ensures all tools are ready for the workflow.",
        6: "### Step 6: Load the Dataset\n**Why:** We read the Excel file `customer_churn_dataset.xlsx` into a variable named `df` (DataFrame). This step loads the data from the disk into the memory so we can analyze and manipulate it.",
        7: "### Step 7: Initial Inspection (Head)\n**Why:** We use `.head()` to display the first 5 rows. This allows us to visually verify that the data loaded correctly and gives us a first look at the column names (e.g., `age`, `churn`) and the type of data they contain (numbers vs text).",
        8: "### Step 8: check Data Info\n**Why:** We run `.info()` to get a technical summary of the DataFrame. We need to know:\n1. The total number of rows (entries).\n2. The data type of each column (`int`, `float`, `object`).\n3. Whether there are any null (missing) values.",
        9: "### Step 9: Count Missing Values\n**Why:** We detected potential missing values in the previous step. Now we sum the nulls (`isnull().sum()`) and sort them to see exactly which columns are incomplete. This is critical because most machine learning models cannot handle missing data.",
        10: "### Step 10: Calculate Missing Percentage\n**Why:** Knowing the *count* isn't enough; we need the *percentage* to decide the strategy. \n- If missingness is low (<5%), we might drop rows.\n- If high, we must impute (fill) them.\n- If extremely high (>50%), we might drop the column.",
        11: "### Step 11: Feature Segmentation\n**Why:** We manually separate our features into two lists: `numerical_features` (like `age`, `charges`) and `categorical_features` (like `contract_type`, `internet_service`).\n**Reason:** These two types require different preprocessing techniques (Scaling vs Encoding).",
        12: "### Step 12: Statistical Summary\n**Why:** We use `.describe()` on the numerical columns. This gives us the mean, min, max, and standard deviation.\n**Reason:** We need to check the *scale* of the data. For example, if `total_charges` is in thousands and `dependents` is single digits, we confirm the need for **Scaling** so the larger numbers don't dominate the model.",
        13: "### Step 13: Check Target Balance\n**Why:** We count the values of the `churn` column (1 vs 0).\n**Reason:** We need to know if the classes are balanced. If 90% of customers don't churn, a model could get 90% accuracy by guessing \"No Churn\" every time. This is called the \"Accuracy Paradox.\"",
        14: "### Step 14: Import Preprocessing Classes\n**Why:** We prepare the specific transformation classes (`OneHotEncoder`, `StandardScaler`) that we will use in our automated pipeline.",
        15: "### Step 15: Construct Preprocessing Pipelines\n**Why:** We build automated processing rules:\n1. **Numeric Pipeline:** First fill missing values (`SimpleImputer` median), then scale (`StandardScaler`).\n2. **Categorical Pipeline:** Fill missing values (`SimpleImputer` most_frequent), then convert text to binary vectors (`OneHotEncoder`).\n**Reason:** Pipelines prevent data leakage and ensure the exact same transformations are applied to training and test data.",
        16: "### Step 16: Combine into ColumnTransformer\n**Why:** We use `ColumnTransformer` to apply the Numeric Pipeline *only* to numeric columns and the Categorical Pipeline *only* to categorical columns, combining the results into a single feature set.",
        17: "### Step 17: Drop Irrelevant Column\n**Why:** We remove `customer_id`. IDs are unique to every row and have no predictive power; including them creates noise and overfitting.",
        18: "### Model 1: Logistic Regression\n**Why this model?** Logistic Regression is a linear classifier that estimates the probability of an event (churn).\n- **Pros:** Highly interpretable (coefficients show feature importance), fast training, works well as a baseline.\n- **Cons:** Assumes a linear relationship between features and churn, which might be too simple.\n\n**Configuration:** We use `class_weight='balanced'` explicitly to handle the class imbalance we discovered earlier (75% vs 25%).",
        19: "### Step 18: Train-Test Split\n**Why:** We split the data into Training (80%) and Testing (20%) sets.\n**Reason:** We must evaluate the model on data it has *never seen* to check for overfitting. We use `stratify=y` to ensure the 80/20 split has the same proportion of churners as the original dataset.",
        20: "### Step 19: Train and Evaluate Logistic Regression\n**Why:** We call `.fit()` to train the model on `X_train` and `.predict()` to test it on `X_test`.\nWe evaluate using:\n- **Classification Report:** Shows Precision (accuracy of positive predictions) and Recall (ability to find all positives).\n- **ROC-AUC:** A metric that summarizes how well the model separates the two classes, independent of the threshold.",
        21: "### Step 20: Baseline Accuracy Check\n**Why:** We calculate the simple accuracy score to get a quick \"headline\" number for performance.",
        22: "### Step 21: Install CatBoost\n**Why:** We want to try a more advanced model. `CatBoost` (Categorical Boosting) is a gradient boosting library designed specifically to handle categorical data well.",
        23: "### Step 22: Import CatBoost\n**Why:** Importing the classifier to use in the script.",
        33: "### Model 2: CatBoost Classifier\n**Why this model?** CatBoost uses Gradient Boosting on Decision Trees.\n- **Pros:** \n  1. Handles categorical variables automatically (no need for OneHot encoding).\n  2. Captures complex, non-linear relationships that Logistic Regression misses.\n  3. Usually provides higher accuracy out-of-the-box.\n- **Cons:** Slower to train, more complex (black box).\n\n**Process:**\n1. We fill NaNs with a placeholder string (Required by CatBoost).\n2. We define the categorical feature indices.\n3. We train the model and evaluate it.",
        34: "### Step 24: CatBoost Accuracy Check\n**Why:** We calculate the final accuracy to compare directly against the Logistic Regression baseline."
    }

    # Dictionary of Post-computation deductions (The "Results")
    post_explanations = {
        1: "**Result:** Core libraries installed successfully.",
        2: "**Result:** Scikit-learn installed. We are ready to build models.",
        3: "**Result:** Excel driver installed.",
        4: "**Result:** Driver active.",
        5: "**Result:** All libraries imported. Environment is ready.",
        6: "**Result:** Data loaded. The `df` variable now holds the dataset in memory.",
        7: "**Observation:** The data looks structured. We see `Yes`/`No` columns (binary), text categories (`Payment Method`), and numerical values (`monthly_charges`). The `churn` column is our target (0 or 1).",
        8: "**Deduction:**\n- We have **1000 rows**.\n- **Missing Data Found:** `internet_service` (only 796 non-null), `tech_support` (953), and `monthly_data_gb` (948) have missing values.\n- We will need to fix these before training.",
        9: "**Result:**\n- `internet_service`: 204 missing.\n- `monthly_data_gb`: 52 missing.\n- `tech_support`: 47 missing.\n- Total cleanliness: Most columns are perfect, but these three need attention.",
        10: "**Deduction:**\n- `internet_service` is missing in **20.4%** of cases. This is too much to drop rows; we must impute.\n- `tech_support` is missing ~4.7%, which is manageable.",
        11: "**Result:** Features are now sorted. `numerical_features` holds 8 cols, `categorical_features` holds 6 cols. This helps us automate processing.",
        12: "**Observation:**\n- `total_charges`: max 7991, mean 4026.\n- `dependents`: max 3, mean 0.9.\n- **Scale Difference:** The huge difference in magnitude (thousands vs single digits) confirms that **StandardScaler is mandatory**. Without it, the model would think `total_charges` is 1000x more important than `dependents` just because the number is bigger.",
        13: "**Deduction (Imbalance):**\n- Non-Churn (0): 742\n- Churn (1): 258\n- **Ratio:** roughly 3:1.\n- **Action:** This confirms we need to use `class_weight='balanced'` in our models, otherwise the model might just predict \"0\" for everyone and still be 74% accurate.",
        14: "**Result:** Tools imported.",
        15: "**Result:** Processing logic defined. The pipeline is now a reusable object.",
        16: "**Result:** Full Preprocessor created. It will take raw data and output a clean, scaled, encoded matrix.",
        17: "**Result:** ID removed. Data frame is now purely predictive features.",
        18: "**Result:** Logistic Regression Pipeline created. It includes the preprocessor steps, so we can feed it raw data directly.",
        19: "**Result:** Data split.\n- `X_train`: 800 samples\n- `X_test`: 200 samples\nStratification ensured both sets have similar Churn rates.",
        20: "**Outcome (Logistic Regression):**\n- **Accuracy:** ~76%.\n- **ROC-AUC:** ~0.85. This is decent.\n- **Issue:** Look at the classification report. The model is struggling slightly to separate the classes perfectly, but the balanced weights helped recall. It provides a solid baseline.",
        21: "**Result:** 76% Accuracy.",
        22: "**Result:** CatBoost installed.",
        23: "**Result:** Imported.",
        33: "**Outcome (CatBoost):**\n- **Accuracy:** ~83% (Higher than 76%).\n- **ROC-AUC:** ~0.88 (Higher than 0.85).\n- **Conclusion:** CatBoost significantly **outperforms** Logistic Regression. This suggests that the relationship between customer features and churn is **non-linear**. For example, the effect of `age` on churn might vary depending on `contract_type`, which CatBoost captures but Logistic Regression can miss.",
        34: "**Final Conclusion:** The CatBoost model is the winner with **83.5% accuracy**. We should use this model for production predictions."
    }
    
    # List of titles of markdown cells created by the PREVIOUS execution (to remove them)
    # Heuristic: these were the exact strings I used in valid_titles for add_markdown.py
    # But simpler: I will assume any markdown cell that is NOT in the "original headers" list is one I made.
    
    # Original headers (to keep)
    original_headers = [
        "# Data Preprocessing",
        "### Data Type Categorization",
        "# Data Cleaning",
        "# Base Model Logistic Regression",
        "# CatBoost Classifier",
        "# Data Type Categorization\n" # sometimes it has newline
    ]
    
    def is_original_markdown(source_lines):
        if not source_lines: return False
        content = "".join(source_lines).strip()
        for header in original_headers:
            if content.startswith(header):
                return True
        return False

    new_cells = []
    
    for cell in nb['cells']:
        # If it's a code cell, wrap it with explanations
        if cell['cell_type'] == 'code':
            exec_count = cell.get('execution_count')
            
            # 1. Add Pre-Explanation
            if exec_count in pre_explanations:
                pre_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [line + "\n" for line in pre_explanations[exec_count].split("\n")]
                }
                if pre_cell["source"]: pre_cell["source"][-1] = pre_cell["source"][-1].rstrip("\n")
                new_cells.append(pre_cell)
            
            # 2. Add Code Cell
            new_cells.append(cell)
            
            # 3. Add Post-Explanation
            if exec_count in post_explanations:
                post_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [line + "\n" for line in post_explanations[exec_count].split("\n")]
                }
                if post_cell["source"]: post_cell["source"][-1] = post_cell["source"][-1].rstrip("\n")
                new_cells.append(post_cell)
                
        # If it's a markdown cell, keep it ONLY if it looks like an original header
        elif cell['cell_type'] == 'markdown':
            if is_original_markdown(cell['source']):
                new_cells.append(cell)
            else:
                # This is likely an explanation cell from the previous run.
                # We skip it, effectively replacing it with the new pre/post structure.
                pass
        
        else:
            # Keep raw/other cells
            new_cells.append(cell)

    nb['cells'] = new_cells
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    process_notebook(r"c:/Users/Aln37/Downloads/Churn_Classification/churn.ipynb")
    print("Notebook re-organized successfully.")
