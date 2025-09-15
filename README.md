# Flex Property Pipeline

A scalable system to identify flex industrial properties in Palm Beach County using free data sources.

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment:
   - Copy `.env` and add your MongoDB connection string
   - Update other configuration as needed

## Usage

Run the complete pipeline:
```bash
python main.py
```

Run specific phases:
```bash
python main.py --phases 1 2 3
```

Run in test mode:
```bash
python main.py --test
```

## Project Structure

- `config/` - Configuration files
- `extractors/` - Data extraction modules
- `processors/` - Data processing and scoring
- `database/` - Database management
- `utils/` - Utility functions
- `data/` - Raw and processed data storage
- `logs/` - Application logs
- `tests/` - Unit tests

## See IMPLEMENTATION_GUIDE.md for detailed setup instructions.