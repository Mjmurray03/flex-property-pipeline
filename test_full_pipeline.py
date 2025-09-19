import time
import pandas as pd
from processors.private_property_analyzer import PrivatePropertyAnalyzer

def clean_numeric_column(series):
    """Clean and convert text-based numeric columns to proper numeric values"""
    if series.dtype == 'object':
        cleaned = series.astype(str).str.replace('$', '', regex=False)
        cleaned = cleaned.str.replace(',', '', regex=False)
        cleaned = cleaned.str.replace('%', '', regex=False)
        cleaned = cleaned.str.strip()
        cleaned = cleaned.replace(['N/A', 'n/a', 'NA', 'na', '', 'None', 'none'], None)
        return pd.to_numeric(cleaned, errors='coerce')
    return series

def test_pipeline():
    file_path = r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx'

    print("="*60)
    print("TESTING FULL PIPELINE ON ACTUAL DATA")
    print("="*60)

    # Step 1: Load and analyze
    print("\n1. Loading data...")
    start = time.time()
    analyzer = PrivatePropertyAnalyzer(file_path)
    df = analyzer.load_data()
    print(f"   Loaded {len(df)} properties in {time.time()-start:.2f}s")

    # Step 2: Apply flex filtering
    print("\n2. Applying flex criteria...")
    # Your optimized criteria with proper data conversion
    industrial_mask = df['Property Type'].str.lower().str.contains(
        'industrial|warehouse|distribution|flex', na=False
    )

    # Convert Building SqFt to numeric with proper cleaning
    df['Building SqFt'] = clean_numeric_column(df['Building SqFt'])
    size_mask = df['Building SqFt'] >= 20000

    lot_mask = (df['Lot Size Acres'] >= 0.5) & (df['Lot Size Acres'] <= 20)

    # Additional stringent filter for demo with proper price conversion
    if 'Sold Price' in df.columns:
        df['Sold Price'] = clean_numeric_column(df['Sold Price'])
        value_mask = (df['Sold Price'] >= 150000) & (df['Sold Price'] <= 20000000)  # Increased upper limit
    else:
        value_mask = pd.Series([True] * len(df))  # No filter if column missing

    flex_df = df[industrial_mask & size_mask & lot_mask & value_mask].copy()
    print(f"   Found {len(flex_df)} flex candidates ({len(flex_df)/len(df)*100:.1f}%)")

    # Step 3: Score properties
    print("\n3. Scoring flex properties...")

    # Calculate scores
    flex_df['flex_score'] = 0
    for idx, row in flex_df.iterrows():
        score = 0

        # Building size scoring
        sqft = row['Building SqFt']
        if 20000 <= sqft <= 50000:
            score += 3
        elif 50000 < sqft <= 100000:
            score += 2
        elif sqft > 100000:
            score += 1

        # Property type scoring
        prop_type = str(row['Property Type']).lower()
        if 'flex' in prop_type:
            score += 3
        elif 'warehouse' in prop_type:
            score += 2.5
        elif 'industrial' in prop_type:
            score += 2

        # Lot size scoring
        acres = row['Lot Size Acres']
        if 1 <= acres <= 5:
            score += 2
        elif 5 < acres <= 10:
            score += 1.5

        # Age scoring
        if pd.notna(row.get('Year Built', None)):
            if row['Year Built'] >= 1990:
                score += 1

        flex_df.at[idx, 'flex_score'] = min(score, 10)

    # Sort by score
    flex_df = flex_df.sort_values('flex_score', ascending=False)

    # Step 4: Display results
    print("\n4. Top 10 Flex Properties:")
    print("-" * 60)
    for i, row in flex_df.head(10).iterrows():
        print(f"{row['Property Name'][:40]:<40} Score: {row['flex_score']:.1f}")
        print(f"  {row['City']}, {row['State']} | {row['Building SqFt']:,.0f} sqft")
        print()

    # Step 5: Export test results
    print("5. Exporting results...")
    output_file = r'C:\flex-property-pipeline\data\exports\test_results.xlsx'
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    flex_df.to_excel(output_file, index=False)
    print(f"   Exported to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Properties: {len(df):,}")
    print(f"Flex Candidates: {len(flex_df):,} ({len(flex_df)/len(df)*100:.1f}%)")
    if len(flex_df) > 0:
        print(f"Average Score: {flex_df['flex_score'].mean():.2f}")
        print(f"Score Range: {flex_df['flex_score'].min():.1f} - {flex_df['flex_score'].max():.1f}")

    return flex_df

if __name__ == "__main__":
    results = test_pipeline()