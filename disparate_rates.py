import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_disparate_approval_rates(df, group_col='applicant_race_1'):
    """
    Computes approval rate by group (race, sex, ethnicity, etc.)
    action_taken == 1 is considered approval.
    """
    valid_df = df[df['action_taken'].notna() & df[group_col].notna()]
    grouped = valid_df.groupby(group_col)['action_taken']

    summary = grouped.apply(lambda x: (x == 1).sum() / len(x) * 100).reset_index()
    summary.columns = [group_col, 'ApprovalRatePercent']
    summary = summary.sort_values(by='ApprovalRatePercent', ascending=False)

    return summary

def plot_disparate_approval_rates(summary_df, group_col):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary_df, x=group_col, y='ApprovalRatePercent')
    plt.ylabel('Approval Rate (%)')
    plt.title(f'Approval Rate by {group_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
