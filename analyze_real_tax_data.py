import json
import pandas as pd
from pathlib import Path

def analyze_real_tax_patterns():
    """Analyze real tax data to derive assessment and calculation patterns"""

    scraped_file = Path('scraped_building_data_50_properties.json')
    with open(scraped_file, 'r') as f:
        scraped_data = json.load(f)

    analysis_data = []

    for item in scraped_data:
        if 'raw_areas' in item and item.get('scrape_success'):
            raw = item['raw_areas']
            orig = item.get('original_data', {})

            if raw.get('total tax') and raw.get('ad valorem'):
                # Extract all relevant values
                market_value = orig.get('market_value', 0) or raw.get('total market value', 0)
                assessed_value = raw.get('assessed value', 0)
                taxable_value = raw.get('taxable value', 0)
                ad_valorem = raw.get('ad valorem', 0)
                non_ad_valorem = raw.get('non ad valorem', 0)
                total_tax = raw.get('total tax', 0)
                building_sqft = item.get('building_sqft', 0)

                if market_value and assessed_value and taxable_value and ad_valorem:
                    # Calculate ratios and rates
                    assessment_ratio = assessed_value / market_value if market_value > 0 else 0
                    exemption_amount = assessed_value - taxable_value if assessed_value >= taxable_value else 0
                    exemption_ratio = exemption_amount / assessed_value if assessed_value > 0 else 0
                    effective_tax_rate = total_tax / market_value if market_value > 0 else 0
                    ad_valorem_rate = ad_valorem / taxable_value if taxable_value > 0 else 0
                    non_ad_valorem_per_sqft = non_ad_valorem / building_sqft if building_sqft > 0 else 0

                    analysis_data.append({
                        'parcel_id': item['parcel_id'],
                        'market_value': market_value,
                        'assessed_value': assessed_value,
                        'taxable_value': taxable_value,
                        'exemption_amount': exemption_amount,
                        'ad_valorem': ad_valorem,
                        'non_ad_valorem': non_ad_valorem,
                        'total_tax': total_tax,
                        'building_sqft': building_sqft,
                        'assessment_ratio': assessment_ratio,
                        'exemption_ratio': exemption_ratio,
                        'effective_tax_rate': effective_tax_rate,
                        'ad_valorem_rate': ad_valorem_rate,
                        'non_ad_valorem_per_sqft': non_ad_valorem_per_sqft,
                        'property_use': orig.get('property_use', ''),
                        'municipality': orig.get('municipality', '')
                    })

    df = pd.DataFrame(analysis_data)

    print("REAL TAX DATA ANALYSIS - PALM BEACH COUNTY INDUSTRIAL/FLEX PROPERTIES")
    print("="*80)
    print(f"Sample Size: {len(df)} properties with complete tax data\n")

    print("ASSESSMENT METHODOLOGY:")
    print(f"Average Assessment Ratio (Assessed/Market): {df['assessment_ratio'].mean():.3f}")
    print(f"Assessment Ratio Range: {df['assessment_ratio'].min():.3f} - {df['assessment_ratio'].max():.3f}")
    print(f"Assessment Ratio Std Dev: {df['assessment_ratio'].std():.3f}\n")

    print("EXEMPTION PATTERNS:")
    print(f"Average Exemption Ratio: {df['exemption_ratio'].mean():.3f}")
    print(f"Exemption Ratio Range: {df['exemption_ratio'].min():.3f} - {df['exemption_ratio'].max():.3f}")
    print(f"Properties with exemptions: {(df['exemption_amount'] > 0).sum()}/{len(df)}\n")

    print("AD VALOREM TAX RATES (Millage):")
    # Convert to mills (per $1,000)
    df['millage_rate'] = df['ad_valorem_rate'] * 1000
    print(f"Average Millage Rate: {df['millage_rate'].mean():.3f} mills")
    print(f"Millage Rate Range: {df['millage_rate'].min():.3f} - {df['millage_rate'].max():.3f} mills")
    print(f"Millage Rate Std Dev: {df['millage_rate'].std():.3f} mills\n")

    print("NON-AD VALOREM ANALYSIS:")
    print(f"Average Non-Ad Valorem per Sqft: ${df['non_ad_valorem_per_sqft'].mean():.3f}")
    print(f"Non-Ad Valorem per Sqft Range: ${df['non_ad_valorem_per_sqft'].min():.3f} - ${df['non_ad_valorem_per_sqft'].max():.3f}")

    # Base fee analysis
    df['estimated_base_fee'] = df['non_ad_valorem'] - (df['non_ad_valorem_per_sqft'] * df['building_sqft'])
    print(f"Estimated Average Base Fee: ${df['estimated_base_fee'].mean():.2f}\n")

    print("EFFECTIVE TAX RATES:")
    df['effective_rate_pct'] = df['effective_tax_rate'] * 100
    print(f"Average Effective Rate: {df['effective_rate_pct'].mean():.3f}%")
    print(f"Effective Rate Range: {df['effective_rate_pct'].min():.3f}% - {df['effective_rate_pct'].max():.3f}%\n")

    print("BY MUNICIPALITY:")
    muni_stats = df.groupby('municipality').agg({
        'millage_rate': 'mean',
        'effective_rate_pct': 'mean',
        'assessment_ratio': 'mean'
    }).round(3)
    print(muni_stats)

    # Return key parameters for estimation
    return {
        'assessment_ratio_mean': df['assessment_ratio'].mean(),
        'assessment_ratio_std': df['assessment_ratio'].std(),
        'exemption_ratio_mean': df['exemption_ratio'].mean(),
        'millage_rate_mean': df['millage_rate'].mean(),
        'millage_rate_std': df['millage_rate'].std(),
        'non_ad_valorem_per_sqft_mean': df['non_ad_valorem_per_sqft'].mean(),
        'base_fee_mean': df['estimated_base_fee'].mean(),
        'municipality_rates': df.groupby('municipality')['millage_rate'].mean().to_dict()
    }

if __name__ == "__main__":
    params = analyze_real_tax_patterns()
    print(f"\nDERIVED ESTIMATION PARAMETERS:")
    print(f"Assessment Ratio: {params['assessment_ratio_mean']:.4f} ± {params['assessment_ratio_std']:.4f}")
    print(f"Millage Rate: {params['millage_rate_mean']:.3f} ± {params['millage_rate_std']:.3f} mills")
    print(f"Non-Ad Valorem: ${params['base_fee_mean']:.0f} base + ${params['non_ad_valorem_per_sqft_mean']:.3f}/sqft")