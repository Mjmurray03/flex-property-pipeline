#!/usr/bin/env python3
"""
Analyze enriched_properties collection data quality
"""
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.mongodb_client import get_db_manager

def analyze_enriched_properties():
    """Analyze the enriched_properties collection"""
    
    # Connect to MongoDB
    db_manager = get_db_manager()
    db = db_manager.db
    
    print('MongoDB Data Quality Analysis')
    print('=' * 50)
    
    try:
        # 1. Count total documents
        total_count = db.enriched_properties.count_documents({})
        print(f'1. Total enriched properties: {total_count}')
        
        if total_count == 0:
            print('No documents found in enriched_properties collection')
            return
        
        # 2. Show a sample enriched property with all fields
        print(f'\n2. Sample enriched property:')
        sample = db.enriched_properties.find_one()
        if sample:
            # Convert ObjectId and datetime for JSON serialization
            sample['_id'] = str(sample['_id'])
            if 'processed_at' in sample:
                sample['processed_at'] = sample['processed_at'].isoformat() if isinstance(sample['processed_at'], datetime) else str(sample['processed_at'])
            
            # Show sample with pretty formatting, but limit size
            sample_str = json.dumps(sample, indent=2, default=str)
            if len(sample_str) > 2000:
                # Show first part and structure
                lines = sample_str.split('\n')
                print('\n'.join(lines[:50]))
                print(f'... [truncated, showing first 50 lines of {len(lines)} total lines]')
            else:
                print(sample_str)
        else:
            print('No sample document found')
        
        # 3. Count flex candidates (>= 5)
        candidates_count = db.enriched_properties.count_documents({'flex_score': {'$gte': 5}})
        print(f'\n3. Flex candidates (score >= 5): {candidates_count}')
        
        # 4. Count prime candidates (>= 8)
        prime_count = db.enriched_properties.count_documents({'flex_score': {'$gte': 8}})
        print(f'4. Prime candidates (score >= 8): {prime_count}')
        
        # 5. Calculate average flex_score
        pipeline = [
            {'$group': {
                '_id': None,
                'avg_score': {'$avg': '$flex_score'},
                'min_score': {'$min': '$flex_score'},
                'max_score': {'$max': '$flex_score'},
                'count': {'$sum': 1}
            }}
        ]
        stats = list(db.enriched_properties.aggregate(pipeline))
        if stats:
            stat = stats[0]
            print(f'5. Flex score statistics:')
            print(f'   Average: {stat["avg_score"]:.2f}')
            print(f'   Minimum: {stat["min_score"]}')
            print(f'   Maximum: {stat["max_score"]}')
            print(f'   Total properties: {stat["count"]}')
        else:
            print('5. No statistical data available')
        
        # Additional analysis - distribution by score range
        print(f'\n6. Score distribution:')
        ranges = [
            (0, 2, 'Very Low'),
            (2, 4, 'Low'), 
            (4, 6, 'Medium'),
            (6, 8, 'High'),
            (8, 10, 'Very High')
        ]
        
        for min_score, max_score, label in ranges:
            count = db.enriched_properties.count_documents({
                'flex_score': {'$gte': min_score, '$lt': max_score}
            })
            percentage = (count / total_count * 100) if total_count > 0 else 0
            print(f'   {label} ({min_score}-{max_score}): {count} ({percentage:.1f}%)')
        
        # Show top 5 highest scoring properties
        print(f'\n7. Top 5 highest scoring properties:')
        top_properties = db.enriched_properties.find({}, {
            'parcel_id': 1, 
            'flex_score': 1, 
            'address': 1, 
            'municipality': 1,
            'market_value': 1,
            'property_use': 1
        }).sort('flex_score', -1).limit(5)
        
        for i, prop in enumerate(top_properties, 1):
            print(f'   {i}. Parcel: {prop.get("parcel_id", "N/A")}')
            print(f'      Score: {prop.get("flex_score", "N/A")}')
            print(f'      Address: {prop.get("address", "N/A")}')
            print(f'      Municipality: {prop.get("municipality", "N/A")}')
            print(f'      Value: ${prop.get("market_value", 0):,.0f}')
            print(f'      Use: {prop.get("property_use", "N/A")}')
            print()
        
        # Check for data quality issues
        print(f'8. Data quality checks:')
        
        # Check for None values in critical fields
        none_scores = db.enriched_properties.count_documents({'flex_score': None})
        print(f'   Properties with None flex_score: {none_scores}')
        
        # Check for properties with details vs without
        with_details = db.enriched_properties.count_documents({'source_data.appraiser': {'$ne': None}})
        without_details = total_count - with_details
        print(f'   Properties with appraiser details: {with_details}')
        print(f'   Properties without appraiser details: {without_details}')
        
        # Check for missing market values
        no_value = db.enriched_properties.count_documents({'market_value': {'$in': [None, 0]}})
        print(f'   Properties with missing/zero market value: {no_value}')
        
        print(f'\n9. Collection statistics:')
        collection_stats = db.command("collStats", "enriched_properties")
        print(f'   Documents: {collection_stats.get("count", "N/A")}')
        print(f'   Size (bytes): {collection_stats.get("size", "N/A"):,}')
        print(f'   Average document size: {collection_stats.get("avgObjSize", "N/A")} bytes')
        
    except Exception as e:
        print(f'Error during analysis: {e}')
        import traceback
        traceback.print_exc()
    finally:
        db_manager.close()

if __name__ == "__main__":
    analyze_enriched_properties()