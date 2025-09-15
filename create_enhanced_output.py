#!/usr/bin/env python3
"""
Create Enhanced Output Files from Checkpoint Data
"""

import json
import pandas as pd
import os
from datetime import datetime

def create_enhanced_output():
    """Create enhanced output files from checkpoint data"""

    print("=== CREATING ENHANCED OUTPUT FROM CHECKPOINT DATA ===")

    # Load base properties
    base_df = pd.read_excel('data/exports/complete_flex_properties.xlsx')
    print(f"Loaded {len(base_df)} base properties")

    # Load enhanced data from checkpoint
    with open('enhanced_scraping_checkpoint.json', 'r') as f:
        checkpoint = json.load(f)

    enhanced_data = checkpoint['enhanced_data']
    print(f"Loaded {len(enhanced_data)} enhanced records")

    # Convert enhanced data to DataFrame
    enhanced_df = pd.DataFrame(enhanced_data)

    # Convert parcel_id to string in both dataframes for consistent merging
    base_df['parcel_id'] = base_df['parcel_id'].astype(str)
    enhanced_df['parcel_id'] = enhanced_df['parcel_id'].astype(str)

    # Merge on parcel_id
    final_df = base_df.merge(enhanced_df, on='parcel_id', how='left')
    print(f"Merged to create {len(final_df)} final records")

    # Add calculated fields for properties with enhanced data
    print("Adding calculated fields...")

    # 5-year appreciation
    final_df['5yr_appreciation'] = None
    valid_data = (final_df['market_value_2025'].notna()) & (final_df['market_value_2021'].notna()) & (final_df['market_value_2021'] > 0)
    final_df.loc[valid_data, '5yr_appreciation'] = (
        (final_df.loc[valid_data, 'market_value_2025'] - final_df.loc[valid_data, 'market_value_2021']) /
        final_df.loc[valid_data, 'market_value_2021'] * 100
    ).round(2)

    # Office to warehouse ratio
    final_df['office_warehouse_ratio'] = None
    valid_areas = (final_df['subarea_office_sqft'].notna()) & (final_df['subarea_warehouse_sqft'].notna()) & \
                  ((final_df['subarea_warehouse_sqft'] + final_df['subarea_office_sqft']) > 0)
    final_df.loc[valid_areas, 'office_warehouse_ratio'] = (
        final_df.loc[valid_areas, 'subarea_office_sqft'] /
        (final_df.loc[valid_areas, 'subarea_warehouse_sqft'] + final_df.loc[valid_areas, 'subarea_office_sqft'])
    ).round(3)

    # Tax rate (percentage)
    final_df['tax_rate'] = None
    valid_tax = (final_df['total_annual_tax'].notna()) & (final_df['market_value_2025'].notna()) & (final_df['market_value_2025'] > 0)
    final_df.loc[valid_tax, 'tax_rate'] = (
        final_df.loc[valid_tax, 'total_annual_tax'] / final_df.loc[valid_tax, 'market_value_2025'] * 100
    ).round(3)

    # Sales count
    final_df['sales_count'] = final_df['sales_history'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Create output directory
    output_dir = 'data/exports'
    os.makedirs(output_dir, exist_ok=True)

    # Save enhanced dataset
    excel_path = os.path.join(output_dir, 'complete_flex_properties_ENHANCED.xlsx')
    csv_path = os.path.join(output_dir, 'complete_flex_properties_ENHANCED.csv')

    final_df.to_excel(excel_path, index=False)
    final_df.to_csv(csv_path, index=False)

    print(f"Enhanced dataset saved to:")
    print(f"  Excel: {excel_path}")
    print(f"  CSV: {csv_path}")

    # Create summary report
    enhanced_count = final_df['scrape_success'].sum() if 'scrape_success' in final_df.columns else len(enhanced_data)

    report = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_properties': len(final_df),
        'enhanced_properties': int(enhanced_count),
        'enhancement_rate': f"{(enhanced_count / len(final_df) * 100):.1f}%",
        'data_summary': {}
    }

    # Data field summary
    enhanced_fields = [
        'subarea_warehouse_sqft', 'subarea_office_sqft', 'zoning_code',
        'property_use_code_detail', 'assessed_value_current', 'taxable_value_current',
        'ad_valorem_tax', 'non_ad_valorem_tax', 'total_annual_tax'
    ]

    for field in enhanced_fields:
        if field in final_df.columns:
            non_null_count = final_df[field].notna().sum()
            report['data_summary'][field] = {
                'records_with_data': int(non_null_count),
                'coverage_rate': f"{(non_null_count / len(final_df) * 100):.1f}%"
            }

    # Top properties by various metrics
    try:
        # Properties with highest tax values
        if 'total_annual_tax' in final_df.columns:
            top_tax = final_df.dropna(subset=['total_annual_tax']).nlargest(10, 'total_annual_tax')[
                ['parcel_id', 'address', 'total_annual_tax', 'taxable_value_current']
            ].to_dict('records')
            report['highest_tax_properties'] = top_tax

        # Properties with lowest tax rates
        if 'tax_rate' in final_df.columns:
            lowest_tax_rate = final_df.dropna(subset=['tax_rate']).nsmallest(10, 'tax_rate')[
                ['parcel_id', 'address', 'tax_rate', 'total_annual_tax']
            ].to_dict('records')
            report['lowest_tax_rate_properties'] = lowest_tax_rate

        # Best office/warehouse ratios
        if 'office_warehouse_ratio' in final_df.columns:
            best_ratio = final_df.dropna(subset=['office_warehouse_ratio']).nlargest(10, 'office_warehouse_ratio')[
                ['parcel_id', 'address', 'office_warehouse_ratio', 'subarea_office_sqft', 'subarea_warehouse_sqft']
            ].to_dict('records')
            report['best_office_warehouse_ratio'] = best_ratio

    except Exception as e:
        print(f"Warning: Error creating detailed report sections: {e}")

    # Save report
    report_path = os.path.join(output_dir, 'enhancement_summary_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Summary report saved to: {report_path}")

    # Print comparison summary
    print("\n=== DATA ENHANCEMENT SUMMARY ===")
    print(f"Base properties: {len(base_df)}")
    print(f"Successfully enhanced: {enhanced_count}")
    print(f"Enhancement rate: {(enhanced_count / len(final_df) * 100):.1f}%")

    print("\n=== NEW DATA FIELDS ADDED ===")
    for field in enhanced_fields:
        if field in final_df.columns:
            count = final_df[field].notna().sum()
            print(f"  {field}: {count} properties ({(count/len(final_df)*100):.1f}%)")

    return final_df, report

if __name__ == "__main__":
    create_enhanced_output()