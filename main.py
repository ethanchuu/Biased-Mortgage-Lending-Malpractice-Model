
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from applicationResults import print_action_summary
from county import print_county_summary, get_county_summary
from disparate_rates import compute_disparate_approval_rates, plot_disparate_approval_rates
from race import print_race_codes, print_race_summary, get_race_summary
from modeling import prepare_data, run_logistic_regression, run_decision_tree, run_knn

def main():
    """Main workflow for HMDA data analysis and modeling."""
    # Read data
    hmda = pd.read_csv('datasets/2017nj.csv', low_memory=False, na_values=["NA", "Exempt", ""])
    hmda = hmda.apply(pd.to_numeric, errors='ignore')

    # Drop columns with >50% missing values
    threshold = len(hmda) * 0.5
    hmda = hmda.dropna(thresh=threshold, axis=1)

    # Drop rows missing key variables
    hmda = hmda.dropna(subset=["loan_amount_000s", "applicant_income_000s"])

    print("\n--- Action Summary ---")
    print_action_summary(hmda)
    print("\n--- County Summary ---")
    print_county_summary(hmda, county_column='county_name')
    print("\n--- Race Summary ---")
    print_race_summary(hmda)

    # Uncomment to analyze approval rates by race or sex
    race_summary = compute_disparate_approval_rates(hmda, group_col='applicant_race_1')
    plot_disparate_approval_rates(race_summary, group_col='applicant_race_1')
    #sex_summary = compute_disparate_approval_rates(hmda, group_col='applicant_sex')
    #plot_disparate_approval_rates(sex_summary, group_col='applicant_sex')

    hmda['default_flag'] = (hmda['loan_amount_000s'] > 200)

    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data(hmda, target_col='default_flag')

    # Drop non-numeric columns from features
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])

    print("\n--- Logistic Regression Results ---")
    print("Running Logistic Regression...")
    run_logistic_regression(X_train, X_test, y_train, y_test)
    print("Logistic Regression finished.\n")

    print("\n--- Decision Tree Results ---")
    print("Running Decision Tree...")
    run_decision_tree(X_train, X_test, y_train, y_test)
    print("Decision Tree finished.\n")

    print("\n--- KNN Results (k=5) ---")
    print("Running KNN...")
    run_knn(X_train, X_test, y_train, y_test, k=5)
    print("KNN finished.\n")

if __name__ == "__main__":
    main()
