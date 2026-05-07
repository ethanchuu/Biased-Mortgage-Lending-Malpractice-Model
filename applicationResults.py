'''
1       Loan originated (approved)
2       Application approved but not accepted
3       Application denied
4       Application withdrawn by applicant
5       File closed for incompleteness
6       Purchased loan (not a new application)
7       Preapproval request denied
8       Preapproval request approved but not accepted
'''
def count_loan_originated(df):
    return (df['action_taken'] == 1).sum()

def count_approved_not_accepted(df):
    return (df['action_taken'] == 2).sum()

def count_denied(df):
    return (df['action_taken'] == 3).sum()

def count_withdrawn(df):
    return (df['action_taken'] == 4).sum()

def count_closed_incomplete(df):
    return (df['action_taken'] == 5).sum()

def count_purchased_loans(df):
    return (df['action_taken'] == 6).sum()

def count_preapproval_denied(df):
    return (df['action_taken'] == 7).sum()

def count_preapproval_approved_not_accepted(df):
    return (df['action_taken'] == 8).sum()

def print_action_summary(df):
    action_count = df['action_taken'].notna().sum()
    print()
    print(f"{'Total Rows with action_taken:':<35} {action_count:,}")
    print()

    counts = {
        "Loan Originated": (df['action_taken'] == 1).sum(),
        "Approved Not Accepted": (df['action_taken'] == 2).sum(),
        "Denied": (df['action_taken'] == 3).sum(),
        "Withdrawn": (df['action_taken'] == 4).sum(),
        "Closed Incomplete": (df['action_taken'] == 5).sum(),
        "Purchased Loans": (df['action_taken'] == 6).sum(),
        "Preapproval Denied": (df['action_taken'] == 7).sum(),
        "Preapproval Approved Not Accepted": (df['action_taken'] == 8).sum()
    }

    print(f"{'Action Taken':<35} {'Count':>10}   {'Percent':>8}")
    print("-" * 60)

    for label, count in counts.items():
        pct = (count / action_count) * 100
        print(f"{label:<35} {count:>10,}   {pct:>7.2f}%")