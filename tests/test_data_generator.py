"""
Test Data Generator for Scalable Multi-File Pipeline
Creates various test scenarios including valid, invalid, and corrupted files
"""

import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import random
import string
from datetime import datetime, timedelta


class PipelineTestDataGenerator:
    """Generate test data for pipeline testing"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize test data generator
        
        Args:
            base_dir: Base directory for test files (uses temp dir if None)
        """
        if base_dir is None:
            self.base_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
        else:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.created_files: List[Path] = []
        
        # Property data templates
        self.property_types = ['Single Family', 'Condo', 'Townhouse', 'Multi-Family', 'Commercial']
        self.zoning_codes = ['R1', 'R2', 'R3', 'C1', 'C2', 'M1', 'PUD']
        self.states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        self.cities = {
            'CA': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento'],
            'TX': ['Houston', 'Dallas', 'Austin', 'San Antonio'],
            'FL': ['Miami', 'Tampa', 'Orlando', 'Jacksonville'],
            'NY': ['New York', 'Buffalo', 'Rochester', 'Syracuse'],
            'IL': ['Chicago', 'Springfield', 'Rockford', 'Peoria']
        }
    
    def generate_property_data(self, 
                             num_properties: int = 1000,
                             flex_candidate_ratio: float = 0.15,
                             duplicate_ratio: float = 0.05) -> pd.DataFrame:
        """
        Generate realistic property data
        
        Args:
            num_properties: Number of properties to generate
            flex_candidate_ratio: Ratio of properties that should be flex candidates
            duplicate_ratio: Ratio of duplicate properties to include
            
        Returns:
            DataFrame with property data
        """
        properties = []
        
        for i in range(num_properties):
            state = random.choice(self.states)
            city = random.choice(self.cities.get(state, ['Generic City']))
            
            # Generate address
            street_num = random.randint(100, 9999)
            street_names = ['Main St', 'Oak Ave', 'Pine Rd', 'Elm Dr', 'Maple Ln', 'Cedar Ct']
            address = f"{street_num} {random.choice(street_names)}"
            
            # Generate property characteristics
            property_type = random.choice(self.property_types)
            zoning = random.choice(self.zoning_codes)
            
            # Lot size (acres)
            lot_size = round(random.uniform(0.1, 2.0), 2)
            
            # Building characteristics
            year_built = random.randint(1950, 2020)
            building_size = random.randint(800, 4000)
            
            # Market values
            land_value = lot_size * random.randint(50000, 200000)
            improvement_value = building_size * random.randint(80, 250)
            total_value = land_value + improvement_value
            
            # Create flex candidates based on ratio
            is_flex_candidate = random.random() < flex_candidate_ratio
            if is_flex_candidate:
                # Adjust characteristics to make it a good flex candidate
                lot_size = max(lot_size, 0.25)  # Minimum lot size
                zoning = random.choice(['C1', 'C2', 'M1', 'PUD'])  # Better zoning
                # Lower improvement to land ratio
                improvement_value = min(improvement_value, land_value * 0.5)
                total_value = land_value + improvement_value
            
            property_data = {
                'parcel_id': f"P{i+1:06d}",
                'site_address': address,
                'city': city,
                'state': state,
                'zip_code': f"{random.randint(10000, 99999)}",
                'acres': lot_size,
                'zoning': zoning,
                'property_type': property_type,
                'year_built': year_built,
                'building_size': building_size,
                'improvement_value': improvement_value,
                'land_market_value': land_value,
                'total_market_value': total_value,
                'last_sale_date': self._random_date(),
                'owner_name': self._random_name(),
                'is_flex_candidate': is_flex_candidate
            }
            
            properties.append(property_data)
        
        df = pd.DataFrame(properties)
        
        # Add duplicates
        if duplicate_ratio > 0:
            num_duplicates = int(num_properties * duplicate_ratio)
            duplicate_indices = random.sample(range(len(df)), num_duplicates)
            
            for idx in duplicate_indices:
                # Create duplicate with slight variations
                duplicate = df.iloc[idx].copy()
                duplicate['parcel_id'] = f"DUP_{duplicate['parcel_id']}"
                duplicate['owner_name'] = self._random_name()  # Different owner
                # Slightly different values
                duplicate['total_market_value'] *= random.uniform(0.95, 1.05)
                
                df = pd.concat([df, duplicate.to_frame().T], ignore_index=True)
        
        return df
    
    def _random_date(self) -> str:
        """Generate random date string"""
        start_date = datetime(2015, 1, 1)
        end_date = datetime(2023, 12, 31)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        return random_date.strftime('%Y-%m-%d')
    
    def _random_name(self) -> str:
        """Generate random owner name"""
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Mary']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def create_valid_excel_file(self, 
                               filename: str,
                               num_properties: int = 1000,
                               flex_ratio: float = 0.15) -> Path:
        """
        Create a valid Excel file with property data
        
        Args:
            filename: Name of the Excel file
            num_properties: Number of properties to generate
            flex_ratio: Ratio of flex candidates
            
        Returns:
            Path to created file
        """
        df = self.generate_property_data(num_properties, flex_ratio)
        file_path = self.base_dir / filename
        
        df.to_excel(file_path, index=False)
        self.created_files.append(file_path)
        
        return file_path
    
    def create_empty_excel_file(self, filename: str) -> Path:
        """Create an empty Excel file"""
        file_path = self.base_dir / filename
        
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        empty_df.to_excel(file_path, index=False)
        self.created_files.append(file_path)
        
        return file_path
    
    def create_invalid_format_file(self, filename: str) -> Path:
        """Create Excel file with invalid/missing columns"""
        file_path = self.base_dir / filename
        
        # Create DataFrame with wrong column names
        invalid_df = pd.DataFrame({
            'wrong_column_1': range(100),
            'wrong_column_2': ['data'] * 100,
            'another_wrong_col': np.random.randn(100)
        })
        
        invalid_df.to_excel(file_path, index=False)
        self.created_files.append(file_path)
        
        return file_path
    
    def create_corrupted_excel_file(self, filename: str) -> Path:
        """Create a corrupted Excel file"""
        file_path = self.base_dir / filename
        
        # Write invalid Excel content
        with open(file_path, 'wb') as f:
            f.write(b'This is not a valid Excel file content')
        
        self.created_files.append(file_path)
        return file_path
    
    def create_large_dataset_file(self, 
                                 filename: str,
                                 num_properties: int = 50000) -> Path:
        """Create large dataset for performance testing"""
        file_path = self.base_dir / filename
        
        # Generate large dataset in chunks to avoid memory issues
        chunk_size = 10000
        all_chunks = []
        
        for i in range(0, num_properties, chunk_size):
            chunk_size_actual = min(chunk_size, num_properties - i)
            chunk_df = self.generate_property_data(
                chunk_size_actual, 
                flex_candidate_ratio=0.12
            )
            # Adjust parcel IDs to be unique across chunks
            chunk_df['parcel_id'] = chunk_df['parcel_id'].apply(
                lambda x: f"L{i//chunk_size:02d}_{x}"
            )
            all_chunks.append(chunk_df)
        
        # Combine all chunks
        large_df = pd.concat(all_chunks, ignore_index=True)
        large_df.to_excel(file_path, index=False)
        self.created_files.append(file_path)
        
        return file_path
    
    def create_mixed_quality_files(self, num_files: int = 10) -> List[Path]:
        """
        Create a mix of valid, invalid, and edge case files
        
        Args:
            num_files: Total number of files to create
            
        Returns:
            List of created file paths
        """
        files = []
        
        # 60% valid files
        num_valid = int(num_files * 0.6)
        for i in range(num_valid):
            file_path = self.create_valid_excel_file(
                f"valid_properties_{i+1}.xlsx",
                num_properties=random.randint(500, 2000),
                flex_ratio=random.uniform(0.1, 0.2)
            )
            files.append(file_path)
        
        # 20% files with different regions (for duplicate testing)
        num_regional = int(num_files * 0.2)
        for i in range(num_regional):
            # Create regional data with some overlapping addresses
            df = self.generate_property_data(
                random.randint(300, 1000),
                flex_candidate_ratio=0.15
            )
            # Force some addresses to be the same for duplicate testing
            if len(df) > 10:
                duplicate_addresses = df['site_address'].iloc[:5].tolist()
                df.loc[5:9, 'site_address'] = duplicate_addresses
            
            file_path = self.base_dir / f"regional_properties_{i+1}.xlsx"
            df.to_excel(file_path, index=False)
            self.created_files.append(file_path)
            files.append(file_path)
        
        # 10% invalid files
        num_invalid = int(num_files * 0.1)
        for i in range(num_invalid):
            if i % 2 == 0:
                file_path = self.create_invalid_format_file(f"invalid_{i+1}.xlsx")
            else:
                file_path = self.create_empty_excel_file(f"empty_{i+1}.xlsx")
            files.append(file_path)
        
        # 10% edge cases
        remaining = num_files - len(files)
        for i in range(remaining):
            if i == 0:
                # Very small file
                file_path = self.create_valid_excel_file(
                    "tiny_dataset.xlsx", 
                    num_properties=5
                )
            else:
                # Corrupted file
                file_path = self.create_corrupted_excel_file(f"corrupted_{i}.xlsx")
            files.append(file_path)
        
        return files
    
    def create_test_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Create predefined test scenarios
        
        Args:
            scenario_name: Name of the scenario to create
            
        Returns:
            Dictionary with scenario information and file paths
        """
        scenarios = {
            'basic_processing': {
                'description': 'Basic processing with valid files',
                'files': [
                    self.create_valid_excel_file("basic_1.xlsx", 1000, 0.15),
                    self.create_valid_excel_file("basic_2.xlsx", 800, 0.12),
                    self.create_valid_excel_file("basic_3.xlsx", 1200, 0.18)
                ]
            },
            'error_handling': {
                'description': 'Test error handling with problematic files',
                'files': [
                    self.create_valid_excel_file("good_file.xlsx", 500, 0.15),
                    self.create_invalid_format_file("bad_format.xlsx"),
                    self.create_empty_excel_file("empty_file.xlsx"),
                    self.create_corrupted_excel_file("corrupted.xlsx")
                ]
            },
            'deduplication': {
                'description': 'Test deduplication with overlapping data',
                'files': [
                    self.create_valid_excel_file("region_a.xlsx", 800, 0.15),
                    self.create_valid_excel_file("region_b.xlsx", 600, 0.12)
                ]
            },
            'performance': {
                'description': 'Performance testing with large datasets',
                'files': [
                    self.create_large_dataset_file("large_dataset_1.xlsx", 20000),
                    self.create_large_dataset_file("large_dataset_2.xlsx", 15000)
                ]
            },
            'mixed_quality': {
                'description': 'Mixed quality files for comprehensive testing',
                'files': self.create_mixed_quality_files(15)
            }
        }
        
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = scenarios[scenario_name]
        scenario['base_dir'] = self.base_dir
        scenario['total_files'] = len(scenario['files'])
        
        return scenario
    
    def get_scenario_summary(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics for a test scenario"""
        files = scenario['files']
        
        total_properties = 0
        valid_files = 0
        invalid_files = 0
        
        for file_path in files:
            try:
                df = pd.read_excel(file_path)
                if not df.empty and 'site_address' in df.columns:
                    total_properties += len(df)
                    valid_files += 1
                else:
                    invalid_files += 1
            except Exception:
                invalid_files += 1
        
        return {
            'total_files': len(files),
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'total_properties': total_properties,
            'avg_properties_per_file': total_properties / max(valid_files, 1)
        }
    
    def cleanup(self):
        """Clean up all created test files"""
        try:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test directory {self.base_dir}: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()


# Convenience functions
def create_test_scenario(scenario_name: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create a test scenario and return scenario info
    
    Args:
        scenario_name: Name of scenario to create
        base_dir: Base directory for test files
        
    Returns:
        Dictionary with scenario information
    """
    generator = PipelineTestDataGenerator(base_dir)
    return generator.create_test_scenario(scenario_name)


def create_sample_files(num_files: int = 5, base_dir: Optional[Path] = None) -> List[Path]:
    """
    Create sample files for quick testing
    
    Args:
        num_files: Number of files to create
        base_dir: Base directory for files
        
    Returns:
        List of created file paths
    """
    generator = PipelineTestDataGenerator(base_dir)
    return generator.create_mixed_quality_files(num_files)


if __name__ == "__main__":
    # Example usage
    with PipelineTestDataGenerator() as generator:
        # Create basic test scenario
        scenario = generator.create_test_scenario('basic_processing')
        print(f"Created scenario: {scenario['description']}")
        print(f"Files: {len(scenario['files'])}")
        
        # Get summary
        summary = generator.get_scenario_summary(scenario)
        print(f"Summary: {summary}")