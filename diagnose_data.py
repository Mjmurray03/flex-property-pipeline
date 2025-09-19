from processors.private_property_analyzer import PrivatePropertyAnalyzer
import pandas as pd

file_path = r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx'

# Initialize analyzer
analyzer = PrivatePropertyAnalyzer(file_path)
df = analyzer.load_data()

print("DATA DIAGNOSTICS")
print("="*50)

# Check Property Type values
print("\nProperty Type Analysis:")
print(df['Property Type'].value_counts().head(10))

# Check Building SqFt values
print("\nBuilding SqFt Analysis:")
df['Building SqFt'] = pd.to_numeric(df['Building SqFt'], errors='coerce')
sqft_stats = df['Building SqFt'].describe()
print(sqft_stats)

print(f"\nBuilding SqFt null values: {df['Building SqFt'].isna().sum()}")
print(f"Zero values: {(df['Building SqFt'] == 0).sum()}")

# Sample of actual values
print("\nSample Building SqFt values (non-null):")
sample_sqft = df[df['Building SqFt'].notna()]['Building SqFt'].head(20)
print(sample_sqft.tolist())

# Size distribution
print("\nSize Distribution:")
print(f"< 5,000 sqft: {(df['Building SqFt'] < 5000).sum()}")
print(f"5,000-10,000 sqft: {((df['Building SqFt'] >= 5000) & (df['Building SqFt'] < 10000)).sum()}")
print(f"10,000-20,000 sqft: {((df['Building SqFt'] >= 10000) & (df['Building SqFt'] < 20000)).sum()}")
print(f">= 20,000 sqft: {(df['Building SqFt'] >= 20000).sum()}")

# Check lot sizes
print("\nLot Size Analysis:")
print(df['Lot Size Acres'].describe())

# Industrial property types
print("\nIndustrial-related property types:")
industrial_types = df[df['Property Type'].str.lower().str.contains('industrial|warehouse|distribution|flex|manufacturing', na=False)]['Property Type'].value_counts()
print(industrial_types)