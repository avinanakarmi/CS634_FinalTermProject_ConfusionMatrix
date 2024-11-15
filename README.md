# Credit Rating Prediction Project

This project evaluates three machine learning models—Random Forest, Decision Tree, and 1D Convolutional Neural Network (CNN)—for classifying an imbalanced credit dataset with high multicollinearity. Key performance metrics are analyzed to identify the best model for accuracy and reliability.

## Objective

The main objective is to:
1. Build and evaluate models to handle data imbalances and feature correlations.
2. Identify the model with the best performance across metrics, including:
   - Accuracy, Precision, Recall, F1 Score, AUC, Balanced Accuracy, TSS, HSS, and Brier Score.

## Project Setup and Execution

### Prerequisites

- **Python version**: 3.12
- **Libraries**: Install the libraries listed in `requirements.txt`.
  ```bash
  pip install -r requirements.txt
  ```

### Running the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/avinanakarmi/nakarmi_avina_finaltermproj.git
   cd nakarmi_avina_finalproj
   ```

2. **Run the Script**:
   Execute the main script to train the models and calculate the performance metrics:
   ```bash
   python nakarmi_avina_finaltermproj.py
   ```

3. **View Results**:
   After execution, the model performance metrics and recommended best model will be displayed.

## Data Exploration and Preprocessing

1. **Data Transformation**: The target variable was transformed for binary classification, creating a new attribute based on quality.
2. **Class Imbalance**: The data contains 19.7% positive and 80.3% negative samples.
3. **Multicollinearity**: Attributes with high correlations were identified, and Variance Inflation Factor (VIF) values were calculated to measure feature interdependencies.

## Model Selection

1. **Random Forest**: Selected for its resilience to multicollinearity and high balanced accuracy.
2. **Decision Tree**: Useful for capturing non-linear feature relationships.
3. **Conv1D CNN**: Capable of learning complex data representations.

## Performance Evaluation

Metrics used to evaluate each model include:
- **True Positive Rate (TPR)**, **True Negative Rate (TNR)**, **Precision**, **F1-Score**
- **Balanced Accuracy**, **True Skill Statistic (TSS)**, **Heidke Skill Score (HSS)**
- **AUC** and **Brier Score (BS)** for probability accuracy

## Best Model

Random Forest achieved the best performance across most metrics, especially in managing false positives and achieving high AUC and balanced accuracy. It is the recommended model for this dataset.
