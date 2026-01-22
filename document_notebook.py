import json

# Define the documentation cells to insert
doc_cells = {
    0: { # Insert at the very top (index 0)
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Notebook Documentation & Analysis\n",
            "This notebook has been documented to explain each step, the models used, and the interpretation of results.\n",
            "\n",
            "### Summary of Models Used\n",
            "1. **Logistic Regression**: A simple linear baseline model.\n",
            "2. **CatBoost Classifier**: A powerful gradient boosting model that handles categorical data well.\n"
        ]
    },
    7: { # After df.head() (which is roughly index 6 in the provided file, let's target indices based on content)
         # I will use content matching in the script instead of hardcoded indices to be safe.
         "text_match": "df.head()",
         "content": [
             "### Output Explanation: Data Snapshot\n",
             "The output above displays the first 5 rows of the dataset. We can observe:\n",
             "- **Target**: `churn` (1 = Left, 0 = Stayed).\n",
             "- **Features**: A mix of numerical (e.g., `monthly_charges`) and categorical (e.g., `payment_method`) data.\n"
         ]
    },
    13: { # After value_counts
          "text_match": "df['churn'].value_counts()",
          "content": [
              "### Output Explanation: Class Imbalance\n",
              "The output shows ~742 retained customers (Class 0) and ~258 churned customers (Class 1).\n",
              "- **Why is this important?**: The dataset is imbalanced. If we don't handle this (using `class_weight='balanced'`), models might just predict '0' for everyone and still get 74% accuracy, essentially learning nothing.\n"
          ]
    },
    21: { # After Logistic Regression Classification Report
         "text_match": "print(\"ROC-AUC:\", roc_auc_score(y_test, y_prob))",
         "content": [
             "### Model Analysis: Logistic Regression Results\n",
             "- **Accuracy**: ~76%\n",
             "- **ROC-AUC**: 0.85\n",
             "- **Interpretation**: The model is decent but not great. It correctly identifies 74% of churners (Recall 0.74). The 'Balanced' weighting helped it find these churners.\n",
             "- **Pros**: Simple, interpretable, fast.\n",
             "- **Cons**: Cannot capture complex, non-linear patterns (e.g., the interaction between age and contract type).\n"
         ]
    },
    33: { # After CatBoost Results
          "text_match": "print(\"ROC-AUC:\", roc_auc_score(y_test2, y_prob2))",
          "content": [
              "### Model Analysis: CatBoost Results\n",
              "- **Accuracy**: ~83.5% (Improved)\n",
              "- **ROC-AUC**: 0.89 (Stronger)\n",
              "- **Why it's better**: CatBoost uses decision trees to find complex rules and handles the categorical nature of 'Contract' and 'Internet Service' natively.\n",
              "- **Parameter Impact**:\n",
              "  - `iterations=300`: More iterations = more learning (risk of overfitting).\n",
              "  - `learning_rate=0.06`: Controls how fast the model changes its mind.\n",
              "  - `depth=6`: Complexity of each decision tree.\n"
          ]
    }
}

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)

def add_documentation(input_path, output_path):
    nb = load_notebook(input_path)
    new_cells = []
    
    # Add header
    new_cells.append(doc_cells[0])
    
    for cell in nb['cells']:
        new_cells.append(cell)
        
        # Check source for matches to insert doc cells AFTER the current cell
        if cell['cell_type'] == 'code':
            source_text = "".join(cell['source'])
            
            for key, doc in doc_cells.items():
                if "text_match" in doc and doc["text_match"] in source_text:
                    # Create the markdown cell
                    md_cell = {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": doc["content"]
                    }
                    new_cells.append(md_cell)
                    break 
                    
    nb['cells'] = new_cells
    save_notebook(output_path, nb)
    print(f"Documentation injected. Saved to {output_path}")

if __name__ == "__main__":
    add_documentation(r"c:\Users\Aln37\Downloads\Churn_Classification\churn.ipynb", 
                      r"c:\Users\Aln37\Downloads\Churn_Classification\churn_documented.ipynb")
