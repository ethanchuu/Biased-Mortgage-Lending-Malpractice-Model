import pandas as pd

def get_county_summary(df, county_column='county_name'):
    valid_df = df[df[county_column].notna()]
    total = len(valid_df)

    county_counts = valid_df[county_column].value_counts().sort_values(ascending=False)
    summary_df = pd.DataFrame({
        'County': county_counts.index,
        'Count': county_counts.values,
        'Percent': (county_counts.values / total) * 100
    })
    return summary_df

def print_county_summary(df, county_column='county_name'):
    summary_df = get_county_summary(df, county_column)
    total = summary_df['Count'].sum()

    print()
    print(f"{'Total Rows with valid counties:':<35} {total:,}\n")
    print(f"{'County':<30} {'Count':>10}   {'Percent':>8}")
    print("-" * 55)

    for _, row in summary_df.iterrows():
        print(f"{row['County']:<30} {row['Count']:>10,}   {row['Percent']:>7.2f}%")