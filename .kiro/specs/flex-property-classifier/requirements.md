# Requirements Document

## Introduction

This feature creates a specialized classifier to identify flex industrial properties from private property datasets using specific criteria including building size, property type, lot size, and multi-tenant suitability. The classifier applies a scoring algorithm to rank properties by their flex potential and exports results for investment analysis.

## Requirements

### Requirement 1

**User Story:** As a real estate investor, I want to classify properties as flex industrial candidates using specific size and type criteria, so that I can identify investment opportunities that match flex property characteristics.

#### Acceptance Criteria

1. WHEN analyzing properties THEN the system SHALL filter for industrial property types using keywords: 'industrial', 'warehouse', 'distribution', 'flex', 'manufacturing', 'storage', 'logistics', 'light industrial'
2. WHEN filtering by building size THEN the system SHALL only include properties with Building SqFt ≥ 20,000
3. WHEN filtering by lot size THEN the system SHALL only include properties with lot size between 0.5 and 20 acres
4. WHEN no properties meet the criteria THEN the system SHALL report zero flex candidates with clear messaging

### Requirement 2

**User Story:** As a real estate investor, I want each flex candidate to receive a numerical score based on multiple factors, so that I can prioritize properties with the highest flex potential.

#### Acceptance Criteria

1. WHEN calculating flex scores THEN the system SHALL use a 0-10 point scale with the following criteria:
   - Building size scoring: 20k-50k sqft = 3 points, 50k-100k = 2 points, 100k-200k = 1 point
   - Property type scoring: 'flex' = 3 points, 'warehouse'/'distribution' = 2.5 points, 'light industrial' = 2 points, 'industrial' = 1.5 points
   - Lot size scoring: 1-5 acres = 2 points, 5-10 acres = 1.5 points, 0.5-1 or 10-20 acres = 1 point
   - Age/condition scoring: built ≥1990 = 1 point, built ≥1980 = 0.5 points
   - Occupancy bonus: <100% occupied = 1 point
2. WHEN calculating scores THEN the system SHALL cap the maximum score at 10 points
3. WHEN displaying results THEN the system SHALL sort properties by flex score in descending order
4. WHEN a scoring factor is missing data THEN the system SHALL assign 0 points for that factor and continue

### Requirement 3

**User Story:** As a real estate investor, I want to export flex candidates with key property details and scores, so that I can analyze and share results with stakeholders.

#### Acceptance Criteria

1. WHEN exporting results THEN the system SHALL include columns: Property Name, Property Type, Address, City, State, County, Building SqFt, Lot Size Acres, Year Built, Zoning Code, Sale Date, Sold Price, Sold Price/SqFt, Owner Name, Occupancy, flex_score
2. WHEN exporting THEN the system SHALL save results to Excel format at 'data/exports/private_flex_candidates.xlsx'
3. WHEN exporting THEN the system SHALL only include columns that exist in the source dataset
4. WHEN export is complete THEN the system SHALL report the number of candidates exported and file location

### Requirement 4

**User Story:** As a real estate investor, I want to see analysis statistics and top candidates, so that I can quickly assess the quality and quantity of flex opportunities in the dataset.

#### Acceptance Criteria

1. WHEN analysis is complete THEN the system SHALL display total number of flex candidates found
2. WHEN analysis is complete THEN the system SHALL show average flex score across all candidates
3. WHEN analysis is complete THEN the system SHALL display the score range (minimum to maximum)
4. WHEN requesting top candidates THEN the system SHALL provide a configurable number of highest-scoring properties (default 100)

### Requirement 5

**User Story:** As a real estate investor, I want the classifier to handle data quality issues gracefully, so that analysis can complete even with incomplete or inconsistent data.

#### Acceptance Criteria

1. WHEN required columns are missing THEN the system SHALL skip those criteria and continue with available data
2. WHEN data types are inconsistent THEN the system SHALL attempt conversion and log warnings for failed conversions
3. WHEN encountering null values in scoring factors THEN the system SHALL treat them as 0 points for that factor
4. WHEN analysis encounters errors THEN the system SHALL log detailed error information and continue processing remaining properties