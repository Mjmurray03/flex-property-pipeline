from processors.private_property_analyzer import PrivatePropertyAnalyzer
import pandas as pd

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

file_path = r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx'

# Initialize analyzer
analyzer = PrivatePropertyAnalyzer(file_path)
df = analyzer.load_data()

print(f"Total properties: {len(df)}")

# Test industrial filtering
industrial_keywords = ['industrial', 'warehouse', 'distribution', 'flex',
                      'manufacturing', 'storage', 'logistics']

industrial_mask = df['Property Type'].str.lower().str.contains(
    '|'.join(industrial_keywords), na=False
)
industrial_count = industrial_mask.sum()
print(f"Industrial properties: {industrial_count}")

# Test size filtering (>=20,000 sqft) with proper conversion
df['Building SqFt'] = clean_numeric_column(df['Building SqFt'])
size_mask = df['Building SqFt'] >= 20000
size_count = size_mask.sum()
print(f"Properties >=20,000 sqft: {size_count}")

# Test lot size filtering (0.5-20 acres)
lot_mask = (df['Lot Size Acres'] >= 0.5) & (df['Lot Size Acres'] <= 20)
lot_count = lot_mask.sum()
print(f"Properties with 0.5-20 acres: {lot_count}")

# Combined criteria for flex properties
flex_mask = industrial_mask & size_mask & lot_mask
flex_count = flex_mask.sum()
print(f"\nFlex candidates: {flex_count} ({flex_count/len(df)*100:.1f}%)")

# Show sample
if flex_count > 0:
    print("\nSample flex properties:")
    sample = df[flex_mask].head(5)
    for _, row in sample.iterrows():
        print(f"  - {row['Property Name']}: {row['Building SqFt']:,.0f} sqft in {row['City']}")
else:
    print("\nNo properties meet all criteria. Showing breakdown:")
    print(f"  Industrial properties: {industrial_count}")
    print(f"  Size >=20K sqft: {size_count}")
    print(f"  Lot 0.5-20 acres: {lot_count}")
    print(f"  Industrial + Size: {(industrial_mask & size_mask).sum()}")
    print(f"  Industrial + Lot: {(industrial_mask & lot_mask).sum()}")
    print(f"  Size + Lot: {(size_mask & lot_mask).sum()}")