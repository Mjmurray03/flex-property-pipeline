from processors.private_property_analyzer import PrivatePropertyAnalyzer
import pandas as pd
import re

def clean_numeric_column(series):
    """Clean and convert text-based numeric columns to proper numeric values"""
    if series.dtype == 'object':
        # Remove common formatting characters
        cleaned = series.astype(str).str.replace('$', '', regex=False)
        cleaned = cleaned.str.replace(',', '', regex=False)
        cleaned = cleaned.str.replace('%', '', regex=False)
        cleaned = cleaned.str.strip()
        # Replace common non-numeric indicators
        cleaned = cleaned.replace(['N/A', 'n/a', 'NA', 'na', '', 'None', 'none'], None)
        return pd.to_numeric(cleaned, errors='coerce')
    return series

def analyze_and_fix_data(file_path):
    """Analyze and demonstrate proper data conversion for all numeric columns"""

    # Initialize analyzer
    analyzer = PrivatePropertyAnalyzer(file_path)
    df = analyzer.load_data()

    print("FIXING DATA CONVERSION ISSUES")
    print("="*50)

    # Identify columns that should be numeric but are stored as text
    numeric_columns = [
        'Building SqFt', 'Loan Amount', 'Interest Rate', 'Sold Price',
        'Lot Size Acres', 'Number of Units', 'Year Built', 'Occupancy',
        'Lease Rate', 'Annual Tax Bill', 'Improvement Value', 'Land Value',
        'Total Parcel Value', 'Sold Price/ SqFt', 'Sold Price/ Acre'
    ]

    print(f"Processing {len(numeric_columns)} potentially numeric columns...")

    for col in numeric_columns:
        if col in df.columns:
            print(f"\nProcessing '{col}':")

            # Show sample raw values
            sample_raw = df[col].dropna().head(5).tolist()
            print(f"  Raw samples: {sample_raw}")

            # Convert to numeric
            df[col + '_cleaned'] = clean_numeric_column(df[col])

            # Show conversion results
            valid_count = df[col + '_cleaned'].notna().sum()
            print(f"  Successfully converted: {valid_count}/{len(df)} values")

            if valid_count > 0:
                stats = df[col + '_cleaned'].describe()
                print(f"  Range: {stats['min']:,.0f} - {stats['max']:,.0f}")
                print(f"  Average: {stats['mean']:,.0f}")

    return df

if __name__ == "__main__":
    file_path = r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx'
    fixed_df = analyze_and_fix_data(file_path)