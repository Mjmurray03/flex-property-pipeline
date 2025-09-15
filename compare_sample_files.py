#!/usr/bin/env python3
"""
Compare Sample Files - Show exact differences between base and enhanced
"""

import pandas as pd

def compare_sample_files():
    """Compare the two sample files and show differences"""

    print("=== SAMPLE FILES COMPARISON ===")

    # Load both sample files
    base_df = pd.read_excel('data/samples/sample_base_properties.xlsx')
    enhanced_df = pd.read_excel('data/samples/sample_enhanced_properties.xlsx')

    print(f"\nBASE SAMPLE: {len(base_df)} properties, {len(base_df.columns)} columns")
    print(f"ENHANCED SAMPLE: {len(enhanced_df)} properties, {len(enhanced_df.columns)} columns")
    print(f"ADDITIONAL FIELDS: {len(enhanced_df.columns) - len(base_df.columns)} new columns")

    print(f"\nBASE DATASET COLUMNS ({len(base_df.columns)}):")
    for i, col in enumerate(base_df.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nENHANCED DATASET - NEW COLUMNS ({len(enhanced_df.columns) - len(base_df.columns)}):")
    new_columns = [col for col in enhanced_df.columns if col not in base_df.columns]
    for i, col in enumerate(new_columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nCOMPLETE ENHANCED DATASET COLUMNS ({len(enhanced_df.columns)}):")
    for i, col in enumerate(enhanced_df.columns, 1):
        marker = "NEW" if col in new_columns else "   "
        print(f"  {i:2d}. {marker} {col}")

    # Show sample data statistics
    print(f"\nDATA COVERAGE IN ENHANCED SAMPLE:")

    coverage_stats = []
    for col in new_columns:
        if col in enhanced_df.columns:
            non_null_count = enhanced_df[col].notna().sum()
            coverage_pct = (non_null_count / len(enhanced_df)) * 100
            coverage_stats.append((col, non_null_count, coverage_pct))

    coverage_stats.sort(key=lambda x: x[2], reverse=True)

    for col, count, pct in coverage_stats:
        print(f"  {col:<25} {count:2d}/{len(enhanced_df)} properties ({pct:5.1f}%)")

    # Show flex score distribution
    print(f"\nFLEX SCORE DISTRIBUTION:")
    score_counts = enhanced_df['flex_score'].value_counts().sort_index(ascending=False)
    for score, count in score_counts.items():
        print(f"  Score {score}: {count} properties")

    # Show examples of enhanced data
    print(f"\nSAMPLE ENHANCED PROPERTY DATA:")
    print("  First property with complete enhanced data:")

    # Find a property with most enhanced data fields
    enhanced_property = None
    max_fields = 0

    for idx, row in enhanced_df.iterrows():
        fields_with_data = sum(1 for col in new_columns if pd.notna(row[col]) and row[col] != '')
        if fields_with_data > max_fields:
            max_fields = fields_with_data
            enhanced_property = row

    if enhanced_property is not None:
        print(f"  Address: {enhanced_property['address']}")
        print(f"  Flex Score: {enhanced_property['flex_score']}")
        print(f"  Enhanced fields ({max_fields}/{len(new_columns)}):")

        for col in new_columns:
            value = enhanced_property[col]
            if pd.notna(value) and value != '':
                if col in ['total_annual_tax', 'improvement_value', 'land_value']:
                    print(f"      {col:<25} ${value:,.0f}")
                elif col in ['tax_rate_percent', '5yr_appreciation_percent', 'office_warehouse_ratio']:
                    print(f"      {col:<25} {value:.2f}%")
                elif col in ['subarea_warehouse_sqft', 'subarea_office_sqft']:
                    print(f"      {col:<25} {value:,.0f} sqft")
                else:
                    print(f"      {col:<25} {value}")

if __name__ == "__main__":
    compare_sample_files()