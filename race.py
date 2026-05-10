import pandas as pd

RACE_CODES = {
    1: 'American Indian or Alaska Native',
    2: 'Asian',
    3: 'Black or African American',
    4: 'Native Hawaiian or Other Pacific Islander',
    5: 'White',
    6: 'Information Not Provided',
    7: 'Not Applicable',
    8: 'No Co-Applicant'
}

def map_race_column(df, col='applicant_race_1'):
    """
    Replaces numeric race codes with human-readable labels.
    """
    df[col] = df[col].map(RACE_CODES).astype(str)
    return df

def print_race_codes():
    print("Race Code Mapping:")
    for code, label in RACE_CODES.items():
        print(f"  {code}: {label}")

def get_race_summary(df, col='applicant_race_1'):
    """
    Returns summary DataFrame with count and percent of each race group.
    """
    valid_df = df[df[col].notna()]
    total = len(valid_df)
    race_counts = valid_df[col].value_counts().sort_values(ascending=False)
    summary_df = pd.DataFrame({
        'Race': race_counts.index,
        'Count': race_counts.values,
        'Percent': (race_counts.values / total) * 100
    })
    return summary_df

def print_race_summary(df, col='applicant_race_1'):
    df = map_race_column(df, col)  # Ensure mapping always happens
    summary = get_race_summary(df, col)
    total = summary['Count'].sum()

    print()
    print(f"{'Total Rows with valid race codes:':<35} {total:,}\n")
    print(f"{'Race':<40} {'Count':>10}   {'Percent':>8}")
    print("-" * 65)

    for _, row in summary.iterrows():
        print(f"{row['Race']:<40} {row['Count']:>10,}   {row['Percent']:>7.2f}%")
