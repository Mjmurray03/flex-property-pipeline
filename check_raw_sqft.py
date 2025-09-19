from processors.private_property_analyzer import PrivatePropertyAnalyzer
import pandas as pd

file_path = r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx'

# Initialize analyzer
analyzer = PrivatePropertyAnalyzer(file_path)
df = analyzer.load_data()

print("RAW BUILDING SQFT VALUES")
print("="*40)

# Check raw values without conversion
print("Sample raw Building SqFt values:")
sample_raw = df['Building SqFt'].head(20).tolist()
for i, val in enumerate(sample_raw):
    print(f"{i+1}: '{val}' (type: {type(val)})")

print("\nUnique Building SqFt values (first 50):")
unique_vals = df['Building SqFt'].unique()[:50]
for val in unique_vals:
    print(f"'{val}' (type: {type(val)})")

# Check for any patterns
print(f"\nTotal unique values: {len(df['Building SqFt'].unique())}")
print(f"Value counts:")
print(df['Building SqFt'].value_counts().head(10))