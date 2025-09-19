import os
import time
import pandas as pd
from datetime import datetime
import json

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

def run_demo():
    """Run a live demo of the flex property pipeline"""

    print("\n" + "="*70)
    print(" FLEX PROPERTY PIPELINE - LIVE DEMO ".center(70))
    print("="*70)

    # Demo configuration - using actual data file
    demo_config = {
        'input_file': r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx',
        'output_file': r'C:\flex-property-pipeline\demo\output\flex_properties_demo.xlsx',
        'min_flex_score': 4.0,
        'max_workers': 4,
        'batch_size': 10
    }

    print("\nDemo Configuration:")
    for key, value in demo_config.items():
        print(f"  {key}: {value}")

    input("\nPress Enter to start processing...")

    # Initialize pipeline
    print("\nVerifying data file...")
    input_file = demo_config['input_file']

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return

    print(f"Found data file: {os.path.basename(input_file)}")

    # Process files
    print("\nProcessing files...")
    start_time = time.time()

    # Load and process data
    from processors.private_property_analyzer import PrivatePropertyAnalyzer
    analyzer = PrivatePropertyAnalyzer(input_file)
    df = analyzer.load_data()

    # Apply filtering with proper data conversion
    industrial_mask = df['Property Type'].str.lower().str.contains(
        'industrial|warehouse|distribution|flex', na=False
    )

    # Convert Building SqFt to numeric with proper cleaning
    df['Building SqFt'] = clean_numeric_column(df['Building SqFt'])
    size_mask = df['Building SqFt'] >= 20000

    lot_mask = (df['Lot Size Acres'] >= 0.5) & (df['Lot Size Acres'] <= 20)

    # Convert Sold Price to numeric with proper cleaning
    if 'Sold Price' in df.columns:
        df['Sold Price'] = clean_numeric_column(df['Sold Price'])
        value_mask = (df['Sold Price'] >= 150000) & (df['Sold Price'] <= 50000000)  # Increased upper limit
    else:
        value_mask = pd.Series([True] * len(df))

    flex_df = df[industrial_mask & size_mask & lot_mask & value_mask].copy()

    # Score properties
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

    # Filter by minimum score
    results = flex_df[flex_df['flex_score'] >= demo_config['min_flex_score']]

    elapsed = time.time() - start_time

    # Show results
    print(f"\nProcessing complete in {elapsed:.1f} seconds")
    print(f"   Processed {len(df)} properties")

    # Filter for flex candidates
    flex_candidates = results
    print(f"   Found {len(flex_candidates)} flex candidates")

    # Display top properties
    print("\nTOP 5 FLEX PROPERTIES:")
    print("-" * 70)

    if not flex_candidates.empty:
        flex_candidates = flex_candidates.sort_values('flex_score', ascending=False)
        for i, row in flex_candidates.head(5).iterrows():
            print(f"\n{row['Property Name']}")
            print(f"  {row['Address']}, {row['City']}, {row['State']}")
            print(f"  {row['Building SqFt']:,.0f} sqft | Built {row.get('Year Built', 'N/A')}")
            print(f"  Flex Score: {row['flex_score']:.1f}/10")

    # Save output
    os.makedirs(os.path.dirname(demo_config['output_file']), exist_ok=True)
    flex_candidates.to_excel(demo_config['output_file'], index=False)

    # Generate report
    print("\nGenerating reports...")
    report_path = r'C:\flex-property-pipeline\demo\reports\demo_report.json'

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_properties': len(df),
        'flex_candidates': len(flex_candidates),
        'pass_rate': f"{len(flex_candidates)/len(df)*100:.1f}%",
        'processing_time': f"{elapsed:.1f} seconds",
        'average_score': flex_candidates['flex_score'].mean() if not flex_candidates.empty else 0
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Report saved to: {report_path}")

    print("\n" + "="*70)
    print(" DEMO COMPLETE ".center(70))
    print("="*70)

if __name__ == "__main__":
    run_demo()