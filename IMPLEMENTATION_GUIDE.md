# FLEX PROPERTY PIPELINE - COMPLETE IMPLEMENTATION GUIDE

## Project Vision
Build a scalable, intelligent system to identify flex industrial properties in Palm Beach County using free data sources. This system will be more valuable than paid services because you'll understand every component and can extend it to any market.

---

## PHASE 1: INITIAL SETUP (Day 1)

### Step 1.1: Clean MongoDB Atlas
**In Claude Code, paste this prompt:**
```
I need to clean up my MongoDB Atlas cluster. My connection string is in the flexfilter-cluster that you can see. Please:
1. Connect to the cluster
2. Show me current collections and their sizes
3. Export the 'parcels' collection to a backup JSON file
4. Drop all existing collections
5. Create these new empty collections: staging_parcels, zoning_data, enriched_properties, flex_candidates
6. Create appropriate indexes for each collection
7. Show me the available storage after cleanup
```

### Step 1.2: Create Project Structure
**In Claude Code, paste this prompt:**
```
Create a new Python project in my current directory with this exact structure:

flex-property-pipeline/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── data_sources.py
├── extractors/
│   ├── __init__.py
│   ├── base_extractor.py
│   ├── gis_extractor.py
│   └── property_extractor.py
├── processors/
│   ├── __init__.py
│   ├── enrichment.py
│   └── flex_scorer.py
├── database/
│   ├── __init__.py
│   ├── mongodb_client.py
│   └── queries.py
├── utils/
│   ├── __init__.py
│   └── logger.py
├── data/
│   ├── raw/
│   └── processed/
├── logs/
├── tests/
├── .env
├── .gitignore
├── requirements.txt
├── README.md
└── main.py

Also initialize a git repository and create a virtual environment.
```

### Step 1.3: Install Dependencies
**In terminal:**
```bash
cd flex-property-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1.4: Configure Environment
1. Copy `.env.example` to `.env`
2. Add your MongoDB connection string
3. Update any API keys or settings

---

## PHASE 2: IMPLEMENT CORE COMPONENTS (Day 2-3)

### Step 2.1: Set Up Database Manager
**Download the `mongodb_client.py` file and place in `database/` folder**

**In Claude Code:**
```
I have a mongodb_client.py file that manages my MongoDB connection with storage limits. Please:
1. Review it and make sure it's properly configured for my 512MB limit
2. Add any missing error handling
3. Create a test script to verify it connects properly
```

### Step 2.2: Implement GIS Extractor
**Download the `gis_extractor.py` file and place in `extractors/` folder**

**Test the GIS extractor:**
```bash
python extractors/gis_extractor.py
```

This should extract industrial parcels and save them to a JSON file.

### Step 2.3: Implement Flex Scorer
**Download the `flex_scorer.py` file and place in `processors/` folder**

### Step 2.4: Set Up Logger
**In Claude Code:**
```
Create a logging utility in utils/logger.py that:
1. Logs to both console and file
2. Uses color coding for different log levels
3. Rotates log files when they reach 10MB
4. Includes timestamp, module name, and line number
```

---

## PHASE 3: RUN INITIAL EXTRACTION (Day 4)

### Step 3.1: Test GIS Extraction
**Run this command:**
```bash
python main_pipeline.py --phases 1 --test
```

This will:
- Extract industrial-zoned parcels from Palm Beach GIS
- Store them in MongoDB
- Create a backup JSON file
- Show you storage usage

### Step 3.2: Monitor Storage
**In Claude Code:**
```
Connect to my MongoDB and show me:
1. Current storage usage
2. Document count in each collection
3. Sample documents from zoning_data collection
```

### Step 3.3: Verify Data Quality
**Create a Jupyter notebook or Python script:**
```python
from database.mongodb_client import get_db_manager

db = get_db_manager()
stats = db.get_collection_stats()
print(stats)

# Get sample parcels
samples = list(db.db.zoning_data.find().limit(5))
for sample in samples:
    print(f"Parcel: {sample['parcel_id']}, Zoning: {sample['zoning']}, Acres: {sample['acres']}")
```

---

## PHASE 4: ENRICHMENT PIPELINE (Day 5-6)

### Step 4.1: Add Property Appraiser Extractor
**In Claude Code:**
```
Create a new file extractors/property_extractor.py that:
1. Takes a list of parcel IDs
2. Fetches detailed property data from Palm Beach Property Appraiser
3. Extracts: owner info, building details, sales history, assessed values
4. Handles rate limiting and retries
5. Returns structured data ready for MongoDB
```

### Step 4.2: Add Permits Extractor
**In Claude Code:**
```
Create extractors/permits_extractor.py that fetches building permits from Palm Beach County for our industrial parcels. Include permit type, date, value, and description.
```

### Step 4.3: Run Enrichment
```bash
python main_pipeline.py --phases 2
```

---

## PHASE 5: ANALYSIS & SCORING (Day 7)

### Step 5.1: Run Full Analysis
```bash
python main_pipeline.py --phases 3
```

### Step 5.2: Export Results
**In Claude Code:**
```
Connect to MongoDB and:
1. Export top 100 flex candidates to CSV
2. Create a summary report with statistics
3. Generate a map-ready GeoJSON file
```

---

## PHASE 6: OPTIMIZATION & SCALING (Week 2)

### Step 6.1: Add More Data Sources
**Business Licenses:**
```python
# Create extractors/business_extractor.py
# Fetch active business licenses to identify multi-tenant properties
```

**Code Violations:**
```python
# Create extractors/violations_extractor.py
# Check for code violations that might indicate property issues
```

### Step 6.2: Implement Caching
**In Claude Code:**
```
Add Redis caching to our pipeline to:
1. Cache GIS queries for 24 hours
2. Cache property details for 7 days
3. Reduce API calls and improve speed
```

### Step 6.3: Create API Endpoint
**In Claude Code:**
```
Create a FastAPI application that:
1. Exposes our flex candidates via REST API
2. Allows filtering by score, location, size
3. Returns detailed property reports
4. Includes basic authentication
```

---

## PHASE 7: MACHINE LEARNING ENHANCEMENT (Week 3)

### Step 7.1: Prepare Training Data
```python
# Manually label 200-300 properties as:
# - Definite flex
# - Possible flex  
# - Not flex
```

### Step 7.2: Train Classifier
**In Claude Code:**
```
Create ml/train_classifier.py that:
1. Loads labeled training data
2. Engineers features from our property data
3. Trains an XGBoost classifier
4. Evaluates performance with cross-validation
5. Saves the model for production use
```

### Step 7.3: Deploy Model
```python
# Integrate classifier into main pipeline
# Re-score all properties with ML model
```

---

## MAINTENANCE & MONITORING

### Daily Tasks
```bash
# Check storage
python scripts/check_storage.py

# Run incremental updates
python main_pipeline.py --incremental --since yesterday
```

### Weekly Tasks
```bash
# Full pipeline run
python main_pipeline.py

# Backup database
mongodump --uri="your_connection_string" --out=backups/

# Generate reports
python scripts/generate_weekly_report.py
```

### Monthly Tasks
- Review and update scoring algorithm
- Add new data sources
- Retrain ML models
- Archive old data

---

## TROUBLESHOOTING

### Storage Issues
```python
# If you hit storage limit:
from database.mongodb_client import get_db_manager
db = get_db_manager()
db.optimize_storage()
db.cleanup_staging(days_old=3)
```

### Rate Limiting
```python
# Adjust in .env:
GIS_RATE_LIMIT=2  # Reduce if getting blocked
```

### Memory Issues
```python
# Process in smaller batches:
BATCH_SIZE=50  # Instead of 100
```

---

## GIT WORKFLOW

### Initial Setup
```bash
git init
git add .
git commit -m "Initial flex property pipeline setup"
git remote add origin https://github.com/yourusername/flex-property-pipeline.git
git push -u origin main
```

### Feature Development
```bash
git checkout -b feature/add-permits-extractor
# Make changes
git add .
git commit -m "Add permits data extractor"
git push origin feature/add-permits-extractor
# Create pull request on GitHub
```

---

## SUCCESS METRICS

### Week 1 Goals
- [ ] Extract 5,000+ industrial parcels
- [ ] Identify 200+ flex candidates
- [ ] Stay under 400MB storage

### Month 1 Goals
- [ ] Full enrichment pipeline operational
- [ ] 90% accuracy in flex identification
- [ ] API serving data to frontend

### Month 3 Goals
- [ ] ML model deployed
- [ ] Expand to 2 more counties
- [ ] 95% accuracy rate

---

## NEXT COUNTY EXPANSION

Once Palm Beach is working:

1. **Broward County:**
   - Update GIS URLs in config
   - Map new zoning codes
   - Run same pipeline

2. **Miami-Dade County:**
   - Adjust for different data structure
   - Add Spanish language support
   - Handle larger dataset (1M+ parcels)

---

## MONETIZATION OPPORTUNITIES

1. **SaaS Product:** $99-499/month for access
2. **API Service:** Charge per API call
3. **Custom Reports:** $500-2000 per report
4. **Data Licensing:** License to brokers/investors
5. **Consulting:** Help others build similar systems

---

## SUPPORT & RESOURCES

- MongoDB Atlas: https://cloud.mongodb.com
- Palm Beach GIS: https://www.pbcgov.org/gis
- Python Async: https://docs.python.org/3/library/asyncio.html
- FastAPI: https://fastapi.tiangolo.com

---

## NOTES

- Always backup before major changes
- Test with small datasets first
- Monitor API rate limits carefully
- Document any custom modifications
- Keep credentials secure and never commit .env

This system will be MORE valuable than CoStar/LoopNet because:
1. You own the data pipeline
2. Completely customizable scoring
3. No monthly fees
4. Can expand to any market
5. Integrate any data source

Start with Phase 1 today and you'll have a working system within a week!
