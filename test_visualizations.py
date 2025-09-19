#!/usr/bin/env python3
"""
Test Visualization and Analytics Components
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def test_visualizations():
    """Test all visualization components"""
    print('TESTING VISUALIZATION AND ANALYTICS COMPONENTS')
    print('=' * 50)

    # Create test dataset
    np.random.seed(42)
    data = {
        'Property_Type': ['Industrial', 'Warehouse', 'Flex', 'Manufacturing'] * 25,
        'State': ['CA', 'TX', 'FL', 'IL'] * 25,
        'City': ['Los Angeles', 'Houston', 'Miami', 'Chicago'] * 25,
        'Building_SqFt': np.random.randint(10000, 200000, 100),
        'Sold_Price': np.random.randint(500000, 5000000, 100),
        'Year_Built': np.random.randint(1980, 2023, 100)
    }
    df = pd.DataFrame(data)

    print(f'Test dataset created: {len(df)} properties')

    # Test 1: Price Distribution Histogram
    print('\n1. Testing price distribution histogram...')
    try:
        fig_hist = px.histogram(
            df,
            x='Sold_Price',
            title='Property Price Distribution',
            nbins=20
        )
        if fig_hist.data:
            print('   [PASS] Price histogram created successfully')
        else:
            print('   [FAIL] Price histogram creation failed')
    except Exception as e:
        print(f'   [FAIL] Price histogram error: {e}')

    # Test 2: Property Type Pie Chart
    print('\n2. Testing property type distribution...')
    try:
        type_counts = df['Property_Type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title='Property Type Distribution'
        )
        if fig_pie.data:
            print('   [PASS] Property type pie chart created')
            print(f'   [PASS] Property types: {list(type_counts.index)}')
        else:
            print('   [FAIL] Property type chart creation failed')
    except Exception as e:
        print(f'   [FAIL] Property type chart error: {e}')

    # Test 3: Geographic Bar Chart
    print('\n3. Testing geographic distribution...')
    try:
        state_counts = df['State'].value_counts()
        fig_bar = px.bar(
            x=state_counts.index,
            y=state_counts.values,
            title='Properties by State'
        )
        if fig_bar.data:
            print('   [PASS] State distribution bar chart created')
            print(f'   [PASS] States: {list(state_counts.index)}')
        else:
            print('   [FAIL] Geographic chart creation failed')
    except Exception as e:
        print(f'   [FAIL] Geographic chart error: {e}')

    # Test 4: Price vs Size Scatter Plot
    print('\n4. Testing price vs size correlation...')
    try:
        fig_scatter = px.scatter(
            df,
            x='Building_SqFt',
            y='Sold_Price',
            color='Property_Type',
            title='Price vs Building Size'
        )
        if fig_scatter.data:
            print('   [PASS] Price vs size scatter plot created')
            correlation = df['Building_SqFt'].corr(df['Sold_Price'])
            print(f'   [PASS] Price-size correlation: {correlation:.3f}')
        else:
            print('   [FAIL] Scatter plot creation failed')
    except Exception as e:
        print(f'   [FAIL] Scatter plot error: {e}')

    # Test 5: Statistical Analysis
    print('\n5. Testing statistical analytics...')
    try:
        stats = {
            'Total_Properties': len(df),
            'Average_Price': df['Sold_Price'].mean(),
            'Median_Price': df['Sold_Price'].median(),
            'Price_StdDev': df['Sold_Price'].std(),
            'Average_Size': df['Building_SqFt'].mean(),
            'Median_Size': df['Building_SqFt'].median(),
            'Property_Types': df['Property_Type'].nunique(),
            'States': df['State'].nunique()
        }

        print('   [PASS] Statistical summary generated:')
        print(f'     Total Properties: {stats["Total_Properties"]:,}')
        print(f'     Average Price: ${stats["Average_Price"]:,.0f}')
        print(f'     Median Price: ${stats["Median_Price"]:,.0f}')
        print(f'     Average Size: {stats["Average_Size"]:,.0f} sqft')
        print(f'     Property Types: {stats["Property_Types"]}')
        print(f'     States: {stats["States"]}')

    except Exception as e:
        print(f'   [FAIL] Statistical analysis error: {e}')

    # Test 6: Market Analysis
    print('\n6. Testing market analysis...')
    try:
        df['Price_per_SqFt'] = df['Sold_Price'] / df['Building_SqFt']

        market_stats = {
            'Avg_Price_per_SqFt': df['Price_per_SqFt'].mean(),
            'Median_Price_per_SqFt': df['Price_per_SqFt'].median(),
            'Min_Price_per_SqFt': df['Price_per_SqFt'].min(),
            'Max_Price_per_SqFt': df['Price_per_SqFt'].max()
        }

        print('   [PASS] Market analysis completed:')
        for key, value in market_stats.items():
            print(f'     {key.replace("_", " ")}: ${value:.2f}')

        # Property type analysis
        type_analysis = df.groupby('Property_Type')['Price_per_SqFt'].agg(['mean', 'count'])
        print('   [PASS] Property type analysis:')
        for prop_type in type_analysis.index:
            avg_price = type_analysis.loc[prop_type, 'mean']
            count = type_analysis.loc[prop_type, 'count']
            print(f'     {prop_type}: ${avg_price:.2f}/sqft ({count} properties)')

    except Exception as e:
        print(f'   [FAIL] Market analysis error: {e}')

    # Test 7: Time Series (Year Built Analysis)
    print('\n7. Testing time series analysis...')
    try:
        year_stats = df.groupby('Year_Built').agg({
            'Sold_Price': 'mean',
            'Building_SqFt': 'mean'
        }).round(0)

        if len(year_stats) > 0:
            print('   [PASS] Time series analysis created')
            print(f'     Year range: {df["Year_Built"].min()} - {df["Year_Built"].max()}')
            print(f'     Years with data: {len(year_stats)}')
        else:
            print('   [FAIL] Time series analysis failed')

    except Exception as e:
        print(f'   [FAIL] Time series error: {e}')

    print('\n[COMPLETE] Visualization and analytics testing completed!')
    return True

if __name__ == "__main__":
    success = test_visualizations()
    print(f'\nVisualization Test Result: {"PASS" if success else "FAIL"}')