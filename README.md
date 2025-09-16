# Flex Property Intelligence Platform

## Advanced Commercial Real Estate Classification System for Palm Beach County

A comprehensive data engineering solution that identifies, classifies, and scores flex properties from 600,000+ Palm Beach County property records using advanced filtering algorithms and public data integration.

---

## 🎯 Executive Summary

This platform demonstrates a production-ready approach to commercial real estate data intelligence, specifically targeting flex properties - the increasingly valuable hybrid industrial/office spaces that represent prime investment opportunities in today's market.

### Key Achievements:
- **611 qualified flex properties** identified from 600,000+ county records
- **342 properties** enhanced with comprehensive tax and ownership data
- **38% of properties** score 10/10 on proprietary scoring algorithm
- **100% public data sources** - no paid APIs required for core functionality

---

## 📊 Dataset Overview

### Core Data Fields (All 611 Properties)
1. **parcel_id** - Unique county identifier
2. **address** - Normalized street address
3. **municipality** - Investment jurisdiction
4. **building_sqft** - Total improved square footage
5. **owner_name** - Current ownership entity
6. **property_use** - Classification code
7. **market_value** - County assessment
8. **flex_score** - Proprietary 1-10 rating
9. **subarea_warehouse_sqft** - Industrial space breakdown
10. **subarea_office_sqft** - Office space breakdown
11. **zoning_code** - Land use permissions
12. **property_use_code_detail** - Granular classification
13. **just_market_value** - Official county valuation
14. **assessed_value_current** - Tax assessment base
15. **exemption_amount** - Tax exemptions (0% for commercial)
16. **taxable_value_current** - Net taxable value

### Enhanced Data Points (342 Properties)
Additional fields extracted from Palm Beach County Property Appraiser official records:
- 5-year appraisal history
- Detailed ownership information
- Tax payment records
- Sale history and deed information
- Building subarea breakdowns

---

## 🔧 Technical Architecture

### Data Pipeline Components

```
Palm Beach County Data Sources
├── Property Appraiser Database (600,000+ records)
├── Tax Roll Files (DR590 format)
├── Building Data (CAMA files)
└── Ownership Records (REC10 format)
          ↓
    Data Processing Layer
    ├── Initial Filtering (20,000+ sqft)
    ├── Classification Algorithm
    ├── Scoring Engine
    └── Data Normalization
          ↓
    Output Formats
    ├── Excel (XLSX) - Business ready
    ├── JSON - API integration
    └── MongoDB - Scalable storage
```

### Technology Stack
- **Python 3.13** - Core processing engine
- **Pandas** - Data manipulation and analysis
- **MongoDB** - Scalable data persistence
- **BeautifulSoup4** - Web scraping capabilities
- **OpenPyXL** - Excel generation

---

## 💰 Value Proposition

### ROI Analysis

| Scenario | Traditional Cost | Platform Cost | ROI |
|----------|-----------------|---------------|-----|
| Single Acquisition (3% broker fee on $3M) | $90,000 | $7,500 | 1,100% |
| Manual Review (1,580 hours @ $75/hr) | $118,500 | $7,500 | 1,480% |
| Market Intelligence Subscription | $24,000/year | $7,500 one-time | 220% first year |

### Time Savings
- **Traditional approach**: 1,580 hours to review 3,160 flex candidates
- **Platform approach**: Instant classification and scoring
- **Time saved**: 99.9% reduction in analysis time

---

## 📁 Repository Structure

```
flex-property-pipeline/
├── data/
│   ├── exports/           # Final datasets (XLSX, JSON)
│   ├── raw/              # Source data from county
│   └── processed/        # Intermediate processing
├── database/
│   └── mongodb_client.py # Database connections
├── scripts/
│   ├── parse_pbcpao_simple.py     # County data parser
│   ├── create_final_clean_sample.py # Sample generator
│   └── real_property_enhancer.py   # Enhancement engine
└── docs/
    └── technical_walkthrough.pdf   # Complete methodology
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
MongoDB (optional for persistence)
4GB RAM minimum
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/flex-property-pipeline.git

# Install dependencies
pip install -r requirements.txt

# Configure MongoDB (optional)
# Update connection string in database/mongodb_client.py
```

### Basic Usage
```python
from parse_pbcpao_simple import SimplePBCPAOParser

# Initialize parser
parser = SimplePBCPAOParser()

# Process county data
enhanced_data = parser.combine_and_export()

# Access results
print(f"Total properties: {len(enhanced_data)}")
print(f"Average market value: ${enhanced_data['market_value'].mean():,.0f}")
```

---

## 📈 Scoring Methodology

### Flex Score Algorithm (1-10 Scale)

The proprietary scoring system evaluates properties across multiple dimensions:

1. **Size Factor** (30% weight)
   - Optimal range: 20,000-50,000 sqft
   - Larger properties score higher for economies of scale

2. **Flexibility Ratio** (25% weight)
   - Office space percentage: 10-30% optimal
   - Indicates true flex potential

3. **Location Quality** (25% weight)
   - Municipality rankings
   - Industrial corridor proximity

4. **Value Metrics** (20% weight)
   - Price per square foot relative to market
   - Tax efficiency ratios

### Score Distribution (611 Properties)
- **Score 10**: 232 properties (38%)
- **Score 8-9**: 189 properties (31%)
- **Score 6-7**: 134 properties (22%)
- **Score 1-5**: 56 properties (9%)

---

## 🔄 Expansion Capabilities

### Additional Counties
The platform architecture supports rapid expansion to additional markets:

- **Broward County** - 12-15 hour implementation
- **Miami-Dade County** - 12-15 hour implementation
- **Orange County** - 12-15 hour implementation
- **Custom Counties** - Available upon request

### Customization Options
- Adjust classification criteria for specific investment thesis
- Modify scoring weights for different strategies
- Add custom data sources and enrichment
- API integration for automated updates

---

## 📞 Professional Services

### Available Packages

#### Palm Beach County Complete Dataset - $7,500
- All 611 qualified flex properties
- 342 enhanced property records
- Complete scoring methodology
- Excel and JSON formats
- 30-day support included

#### County Expansion - $1,875 per county
- Same comprehensive methodology
- 12-15 hour turnaround
- Full data normalization
- Integration with existing dataset

#### Custom Development
- Tailored classification criteria
- Proprietary scoring algorithms
- API integration
- Ongoing data updates

---

## 📝 License & Support

This repository demonstrates production-ready commercial real estate data intelligence capabilities. The methodology and codebase are proprietary and available for licensing.

### Support Channels
- Technical documentation included
- 30-day email support with dataset purchase
- Custom development available

### Data Sources
All data sourced from public records:
- Palm Beach County Property Appraiser
- Palm Beach County Tax Collector
- Florida Department of Revenue

---

## 🤝 Contact

**Michael Murray**
Commercial Real Estate Data Solutions

*Turning public data into private advantage for commercial real estate investors*

---

*This platform represents 60+ hours of development and optimization, providing institutional-quality data intelligence at a fraction of traditional costs.*