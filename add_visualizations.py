
import json
import os

# Content for Visualization Cells

# 1. Imports for Viz
viz_imports_code = """
import matplotlib.pyplot as plt
import seaborn as sns
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
"""

viz_imports_md_why = """### Step: Import Visualization Libraries
**Why do we need this?**
We are importing `matplotlib` and `seaborn`.
- `matplotlib` is the foundation for plotting in Python.
- `seaborn` makes the plots look beautiful and statistical by default.
- We set the figure size to (10, 6) to ensure charts are readable."""

# 2. Target Distribution
viz_target_code = """
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='churn', data=df, palette='viridis')
plt.title('Distribution of Churn (Target Variable)')
plt.xlabel('Churn (0=No, 1=Yes)')
plt.ylabel('Count')

# Add percentages
total = len(df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.show()
"""

viz_target_md_why = """### Visualization 1: Target Class Distribution
**Why is this suitable?**
Before building any model, we must know if our dataset is balanced. If 90% of customers didn't churn, a "dumb" model could guess "No Churn" and be 90% accurate. This chart shows specifically how severe the imbalance is."""


viz_target_md_analysis = """**Graph Explanation & Analysis:**
- The bar chart shows the absolute count of Non-Churners (0) vs Churners (1).
- **Result:** We see a breakdown of approximately **74.2% No Churn** vs **25.8% Churn**.
- **Conclusion:** This confirms a "Class Imbalance". We are correct to use `class_weight='balanced'` in our models later. If we treated them equally, the model would be biased towards the majority class (No Churn)."""

# 3. Correlation Matrix
viz_corr_code = """
plt.figure(figsize=(10, 8))
# Calculate correlation only on numerical columns
corr = df[numerical_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
"""

viz_corr_md_why = """### Visualization 2: Correlation Heatmap
**Why is this suitable?**
Machine learning models (especially Linear Regression) can be confused if two variables mean the same thing (Multicollinearity). This map helps us identify redundant features."""

viz_corr_md_analysis = """**Graph Explanation & Analysis:**
- Dark Red = Strong Positive Correlation. Dark Blue = Strong Negative.
- **Key Finding:** `tenure_months` and `total_charges` have a very high correlation (>0.8).
- **Why?** Because `Total Charges` is basically `Monthly Charges` × `Tenure`.
- **Deduction:** The model might find `total_charges` redundant. Tree-based models (like CatBoost) handle this fine, but for Linear Regression, we sometimes drop one of them (though we kept both here for completeness)."""

# 4. Churn by Contract Type
viz_contract_code = """
plt.figure(figsize=(8, 6))
# Create a cross-tabulation
pd.crosstab(df['contract_type'], df['churn'], normalize='index').plot(kind='bar', stacked=True, color=['#4daf4a', '#e41a1c'], figsize=(8,6))
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Proportion')
plt.legend(title='Churn', labels=['No', 'Yes'], loc='upper right')
plt.xticks(rotation=0)
plt.show()
"""

viz_contract_md_why = """### Visualization 3: Churn by Contract Type
**Why is this suitable?**
"Contract Type" is often a strong predictor. Users on month-to-month contracts can leave anytime, while 2-year contracts are locked in. We use a **Stacked Bar Chart** normalized to 100% to compare the *rate* of churn across groups easily."""

viz_contract_md_analysis = """**Graph Explanation & Analysis:**
- Green = Stayed, Red = Churned.
- **Key Finding:** The "Monthly" bar has a significantly larger Red section than "Two-year".
- **Conclusion:** **Month-to-month customers are the highest risk group.** Longer contracts stabilize the customer base. This is a crucial business insight: incentivizing users to switch to yearly contracts could reduce churn."""

# 5. Feature Importance (CatBoost)
viz_imp_code = """
feature_importance = cat_model.get_feature_importance()
feature_names = X.columns
formatted_fi = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
formatted_fi = formatted_fi.sort_values(by='importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=formatted_fi, palette='viridis')
plt.title('Top 10 Important Features (CatBoost)')
plt.show()
"""

viz_imp_md_why = """### Visualization 4: Feature Importance
**Why is this suitable?**
Our CatBoost model gave us 83% accuracy, but it's a black box. This plot extracts the internal "Gain" score, telling us *which variables actually drove the decision*."""

viz_imp_md_analysis = """**Graph Explanation & Analysis:**
- Longer bars = More influential feature.
- **Result:** You will likely see `contract_type`, `tenure_months`, or `monthly_charges` at the top.
- **Deduction:** If `contract_type` is #1, it confirms our visual hypothesis from earlier. Identifying the top drivers allows marketing teams to focus their efforts (e.g., if `tech_support` is high, maybe improving support reduces churn)."""

# 6. Confusion Matrix
viz_cm_code = """
from sklearn.metrics import ConfusionMatrixDisplay

# Predict class labels
y_pred_cat = cat_model.predict(X_test2)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test2, y_pred_cat, display_labels=['No Churn', 'Churn'], cmap='Blues', ax=ax)
plt.title('Confusion Matrix (CatBoost)')
plt.grid(False)
plt.show()
"""

viz_cm_md_why = """### Visualization 5: Confusion Matrix
**Why is this suitable?**
Accuracy can be misleading. We need to know *how* it fails.
- **False Negative (FN):** Predicted "Safe" but they Churned (Worst case for business - we lost a customer we didn't try to save).
- **False Positive (FP):** Predicted "Churn" but they Stayed (Wasted marketing money)."""

viz_cm_md_analysis = """**Graph Explanation & Analysis:**
- **Diagonal (Dark blue):** Correct predictions.
- **Top Right (False Positives):** Customers we wrongly flagged.
- **Bottom Left (False Negatives):** Customers who slipped away undetected.
- **Deduction:** Look at the Bottom Left number. If it is low, our model is good at "Recall" (catching churners). The `balanced` weighting we used earlier specifically tries to minimize this number."""

def create_cell(cell_type, source):
    lines = source.strip().split("\n")
    # Add newlines back
    lines = [l + "\n" for l in lines]
    # Remove last newline
    if lines: lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        "execution_count": None if cell_type == "code" else None,
        "outputs": []
    }

def process_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    
    # Flags to ensure we only insert once
    eda_inserted = False
    model_viz_inserted = False
    
    for cell in nb['cells']:
        new_cells.append(cell)
        
        # INSERT EDA SECTION
        # We look for the cell where we checked numerical stats (execution_count 12) or target (13)
        # Let's insert after execution_count 13 ("df['churn'].value_counts()")
        # And after its explanation cell
        if not eda_inserted and cell.get('execution_count') == 13:
            # We are at the code cell. The next cell in the loop might be the explanation markdown.
            # We want to insert AFTER the explanation.
            # However, simpler logic: insert it now, and it will appear before the next "Data Cleaning" header
            pass
            
        # Better trigger: Look for the markdown "### Step 13: Check Target Balance"
        # Or just look for the code execution 13 and insert *after* the *next* markdown cell?
        # Let's manually inject into the list logic.
        
    # Re-build list with insertions
    final_cells = []
    i = 0
    while i < len(nb['cells']):
        cell = nb['cells'][i]
        final_cells.append(cell)
        
        # CHECK FOR INSERTION POINTS
        
        # 1. EDA: Insert after Step 13's post-explanation. 
        # Step 13 code has execution_count 13.
        # The next cell is likely the post-explanation markdown.
        if cell.get('cell_type') == 'code' and cell.get('execution_count') == 13:
            # Check if next is markdown explanation
            if i+1 < len(nb['cells']) and nb['cells'][i+1].get('cell_type') == 'markdown':
                final_cells.append(nb['cells'][i+1]) # Add the explanation
                i += 1
            
            # NOW INSERT EDA
            final_cells.append(create_cell("markdown", "# Visual Exploratory Data Analysis (EDA)"))
            
            # Imports
            final_cells.append(create_cell("markdown", viz_imports_md_why))
            final_cells.append(create_cell("code", viz_imports_code))
            
            # Target Viz
            final_cells.append(create_cell("markdown", viz_target_md_why))
            final_cells.append(create_cell("code", viz_target_code))
            final_cells.append(create_cell("markdown", viz_target_md_analysis))
            
            # Corr Viz
            final_cells.append(create_cell("markdown", viz_corr_md_why))
            final_cells.append(create_cell("code", viz_corr_code))
            final_cells.append(create_cell("markdown", viz_corr_md_analysis))
            
            # Contract Viz
            final_cells.append(create_cell("markdown", viz_contract_md_why))
            final_cells.append(create_cell("code", viz_contract_code))
            final_cells.append(create_cell("markdown", viz_contract_md_analysis))
            
            eda_inserted = True

        # 2. Model Viz: Insert after Step 33's post-explanation (CatBoost)
        elif cell.get('cell_type') == 'code' and cell.get('execution_count') == 33:
             # Check if next is markdown explanation
            if i+1 < len(nb['cells']) and nb['cells'][i+1].get('cell_type') == 'markdown':
                final_cells.append(nb['cells'][i+1]) # Add the explanation
                i += 1
                
            # NOW INSERT MODEL VIZ
            final_cells.append(create_cell("markdown", "# Advanced Model Visualization"))
            
            # Feature Importance
            final_cells.append(create_cell("markdown", viz_imp_md_why))
            final_cells.append(create_cell("code", viz_imp_code))
            final_cells.append(create_cell("markdown", viz_imp_md_analysis))
            
            # Confusion Matrix
            final_cells.append(create_cell("markdown", viz_cm_md_why))
            final_cells.append(create_cell("code", viz_cm_code))
            final_cells.append(create_cell("markdown", viz_cm_md_analysis))
            
            model_viz_inserted = True
            
        i += 1

    nb['cells'] = final_cells
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    process_notebook(r"c:/Users/Aln37/Downloads/Churn_Classification/churn.ipynb")
    print("Visualizations added successfully.")
