import pandas as pd
import sys

file_path = r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx'

# Load first 5 rows to check structure
df = pd.read_excel(file_path, nrows=5)

# Expected columns from your specification
expected_columns = [
    'Property Link', 'Property Name', 'Property Type', 'APN',
    'Address', 'Unit', 'City', 'Zip Code', 'State', 'County',
    'Sale Date', 'Sold Price', 'Lease Signed', 'Lease Rate',
    'Building SqFt', 'Number of Units', 'Year Built', 'Lot Size Acres',
    'Owner Name', 'Occupancy', 'Zoning Code'
]

# Check which columns exist
missing = [col for col in expected_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
    print(f"Available columns: {list(df.columns)[:20]}...")
else:
    print("All critical columns found")

print(f"\nData preview:")
print(df[['Property Name', 'Property Type', 'Building SqFt', 'City']].head())