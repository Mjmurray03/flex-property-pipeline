import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import io
from utils.data_type_utils import convert_categorical_to_numeric, setup_logging

# Page configuration
st.set_page_config(
    page_title="Flex Property Filter Dashboard",
    page_icon="[BUILDING]",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(file_path):
    """Load and cache the property data with comprehensive error handling"""
    try:
        # Check if file exists
        import os
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            st.info("Please ensure the file path is correct and the file exists.")
            return pd.DataFrame()
        
        # Try to load from processors module first
        try:
            from processors.private_property_analyzer import PrivatePropertyAnalyzer
            analyzer = PrivatePropertyAnalyzer(file_path)
            df = analyzer.load_data()
        except ImportError:
            # Fallback to direct pandas loading
            try:
                df = pd.read_excel(file_path)
            except Exception as excel_error:
                st.error(f"Error reading Excel file: {str(excel_error)}")
                st.info("Please ensure the file is a valid Excel format (.xlsx or .xls)")
                return pd.DataFrame()
        
        if df.empty:
            st.warning("The loaded file contains no data.")
            return pd.DataFrame()
        
        # Clean numeric columns with error handling
        numeric_columns = [
            'Building SqFt', 'Sold Price', 'Lot Size Acres', 
            'Year Built', 'Loan Amount', 'Interest Rate',
            'Number of Units', 'Lot Size SqFt', 'Occupancy'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = clean_numeric_column(df[col])
                except Exception as clean_error:
                    st.warning(f"Could not clean column '{col}': {str(clean_error)}")
                    continue
        
        return df
    
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        st.info("Please ensure the file path is correct and the file exists.")
        return pd.DataFrame()
    except PermissionError:
        st.error(f"Permission denied accessing file: {file_path}")
        st.info("Please check file permissions and ensure the file is not open in another application.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        st.info("Please check the file format and try again.")
        return pd.DataFrame()

def clean_numeric_column(series):
    """Clean text-based numeric columns"""
    if series.dtype == 'object':
        cleaned = series.astype(str).str.replace('$', '', regex=False)
        cleaned = cleaned.str.replace(',', '', regex=False)
        cleaned = cleaned.str.replace('%', '', regex=False)
        cleaned = cleaned.str.strip()
        cleaned = cleaned.replace(['N/A', 'n/a', 'NA', 'na', '', 'None', 'none'], None)
        return pd.to_numeric(cleaned, errors='coerce')
    return series

def initialize_filter_state():
    """Initialize filter state in session state if not exists"""
    if 'filter_applied' not in st.session_state:
        st.session_state.filter_applied = False
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'filter_params' not in st.session_state:
        st.session_state.filter_params = None

def initialize_upload_state():
    """Initialize file upload state in session state"""
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'upload_status' not in st.session_state:
        st.session_state.upload_status = 'none'  # 'none', 'uploading', 'processing', 'complete', 'error'
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    if 'file_size' not in st.session_state:
        st.session_state.file_size = 0
    if 'upload_timestamp' not in st.session_state:
        st.session_state.upload_timestamp = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0.0
    if 'validation_passed' not in st.session_state:
        st.session_state.validation_passed = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None

def validate_uploaded_file(uploaded_file):
    """Validate uploaded file format and size with comprehensive error handling"""
    if uploaded_file is None:
        return False, "No file uploaded", None
    
    try:
        # Check if file has a name
        if not hasattr(uploaded_file, 'name') or not uploaded_file.name:
            return False, "Invalid file: File name is missing", "file_name_missing"
        
        # Check file extension
        file_parts = uploaded_file.name.lower().split('.')
        if len(file_parts) < 2:
            return False, "Invalid file: No file extension found. Please upload an Excel file (.xlsx or .xls)", "no_extension"
        
        file_extension = file_parts[-1]
        if file_extension not in ['xlsx', 'xls']:
            error_msg = f"Invalid file format: .{file_extension}\n\n"
            error_msg += "**Supported formats:**\n"
            error_msg += "• Excel 2007+ (.xlsx)\n"
            error_msg += "• Excel 97-2003 (.xls)\n\n"
            error_msg += "**How to fix:**\n"
            error_msg += "1. Save your file as Excel format\n"
            error_msg += "2. Use 'Save As' and select Excel Workbook (.xlsx)"
            return False, error_msg, "invalid_format"
        
        # Check if file has content
        if not hasattr(uploaded_file, 'size') or uploaded_file.size == 0:
            return False, "Invalid file: File is empty (0 bytes)", "empty_file"
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB in bytes
        if uploaded_file.size > max_size:
            error_msg = f"File too large: {uploaded_file.size / (1024*1024):.1f}MB\n\n"
            error_msg += "**Maximum allowed size:** 50MB\n\n"
            error_msg += "**How to reduce file size:**\n"
            error_msg += "1. Remove unnecessary columns or rows\n"
            error_msg += "2. Remove formatting and images\n"
            error_msg += "3. Split large datasets into smaller files\n"
            error_msg += "4. Save as .xlsx instead of .xls (usually smaller)"
            return False, error_msg, "file_too_large"
        
        # Basic file content validation
        try:
            # Try to read a small portion to validate it's a valid Excel file
            file_buffer = io.BytesIO(uploaded_file.getvalue())
            pd.read_excel(file_buffer, nrows=1)  # Just read first row to validate
        except Exception as e:
            error_msg = "File appears to be corrupted or not a valid Excel file\n\n"
            error_msg += "**Common causes:**\n"
            error_msg += "• File was not saved properly\n"
            error_msg += "• File is password protected\n"
            error_msg += "• File contains unsupported features\n\n"
            error_msg += "**How to fix:**\n"
            error_msg += "1. Re-save the file as a new Excel workbook\n"
            error_msg += "2. Remove password protection\n"
            error_msg += "3. Copy data to a new Excel file\n\n"
            error_msg += f"**Technical error:** {str(e)}"
            return False, error_msg, "corrupted_file"
        
        return True, "File validation passed", None
    
    except Exception as e:
        return False, f"Unexpected error during file validation: {str(e)}", "validation_error"

def show_error_guidance(error_type, error_message):
    """Show contextual error guidance based on error type"""
    st.error(f"Error: {error_message}")
    
    if error_type == "invalid_format":
        with st.expander("How to convert your file to Excel format"):
            st.markdown("""
            **If you have a CSV file:**
            1. Open the CSV file in Excel
            2. Click 'File' → 'Save As'
            3. Choose 'Excel Workbook (.xlsx)' as the format
            4. Click 'Save'
            
            **If you have a Google Sheets file:**
            1. Open your Google Sheets file
            2. Click 'File' → 'Download' → 'Microsoft Excel (.xlsx)'
            3. Upload the downloaded file
            
            **If you have data in another format:**
            1. Copy your data
            2. Open Excel and paste the data
            3. Save as Excel Workbook (.xlsx)
            """)
    
    elif error_type == "file_too_large":
        with st.expander("How to reduce file size"):
            st.markdown("""
            **Remove unnecessary data:**
            • Delete empty rows and columns
            • Remove columns you don't need for analysis
            • Filter to only include relevant data
            
            **Optimize formatting:**
            • Remove cell formatting, colors, and borders
            • Delete images and charts
            • Remove merged cells
            
            **Split large datasets:**
            • Divide data into multiple files by date, region, etc.
            • Process files separately and combine results
            
            **Use efficient formats:**
            • Save as .xlsx instead of .xls
            • Remove formulas and keep only values
            """)
    
    elif error_type == "corrupted_file":
        with st.expander("How to fix corrupted files"):
            st.markdown("""
            **Try these solutions:**
            1. **Re-save the file:**
               • Open in Excel and save as a new file
               • Use 'Save As' and choose Excel Workbook (.xlsx)
            
            2. **Remove password protection:**
               • If file is password protected, remove protection
               • Save without password
            
            3. **Copy to new file:**
               • Select all data (Ctrl+A)
               • Copy (Ctrl+C)
               • Create new Excel file and paste (Ctrl+V)
               • Save the new file
            
            4. **Check file integrity:**
               • Try opening the file in Excel first
               • Look for any error messages
               • Repair the file using Excel's built-in repair feature
            """)
    
    elif error_type == "empty_file":
        with st.expander("File appears to be empty"):
            st.markdown("""
            **Check your file:**
            • Make sure the file contains data
            • Verify data is in the first worksheet
            • Ensure there are column headers
            
            **Common issues:**
            • Data might be in a different sheet
            • File might have been saved incorrectly
            • Headers might be missing
            """)

def show_troubleshooting_tips():
    """Show general troubleshooting tips"""
    with st.expander("Troubleshooting Tips"):
        st.markdown("""
        **Common Upload Issues:**
        
        **File won't upload:**
        • Check your internet connection
        • Try refreshing the page
        • Make sure file is not open in Excel
        
        **Processing takes too long:**
        • Large files may take several minutes
        • Try reducing file size
        • Check if file has many formulas
        
        **Data looks wrong:**
        • Check column headers match expected names
        • Verify data types (numbers vs text)
        • Look for merged cells or special formatting
        
        **Missing columns:**
        • Required: Property Type, City, State
        • Check spelling and capitalization
        • Use column mapping suggestions
        
        **Still having issues?**
        • Download the sample template
        • Copy your data to the template format
        • Try uploading the template-formatted file
        """)

def show_upload_help():
    """Show comprehensive upload help"""
    st.markdown("### Need Help?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_troubleshooting_tips()
    
    with col2:
        with st.expander("Support Resources"):
            st.markdown("""
            **Quick Solutions:**
            • Download the sample template below
            • Check file format requirements
            • Verify column names match expected format
            
            **Data Preparation Checklist:**
            • File is in Excel format (.xlsx or .xls)
            • File size is under 50MB
            • Data starts from row 1 (no empty rows at top)
            • Column headers are in the first row
            • Required columns are present
            • No merged cells in data area
            • Numbers are formatted as numbers (not text)
            
            **Best Practices:**
            • Use consistent naming conventions
            • Avoid special characters in data
            • Keep one data table per sheet
            • Remove unnecessary formatting
            """)

def generate_filter_summary(df, filtered_df, size_range, lot_range, price_range, year_range, occupancy_range, industrial_keywords, selected_counties, selected_states, use_price_filter, use_year_filter, use_occupancy_filter):
    """Generate comprehensive filter summary with upload metadata"""
    filter_summary = {
        "Data Source": st.session_state.get('uploaded_filename', 'Unknown'),
        "Upload Date": st.session_state.get('upload_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if st.session_state.get('upload_timestamp') else 'Unknown',
        "Processing Time": f"{st.session_state.get('processing_time', 0):.2f}s",
        "Total Properties": len(df),
        "Filtered Properties": len(filtered_df),
        "Pass Rate": f"{len(filtered_df)/len(df)*100:.1f}%",
        "Building Size Range": f"{size_range[0]:,} - {size_range[1]:,} sqft",
        "Lot Size Range": f"{lot_range[0]} - {lot_range[1]} acres",
        "Industrial Keywords": ", ".join(industrial_keywords) if industrial_keywords else "None",
        "Counties Selected": len(selected_counties),
        "States Selected": len(selected_states)
    }
    
    if use_price_filter:
        filter_summary["Price Range"] = f"${price_range[0]:,} - ${price_range[1]:,}"
    
    if use_year_filter:
        filter_summary["Year Range"] = f"{year_range[0]} - {year_range[1]}"
    
    if use_occupancy_filter:
        filter_summary["Occupancy Range"] = f"{occupancy_range[0]}% - {occupancy_range[1]}%"
    
    return filter_summary

def generate_enhanced_csv_export(filtered_df, original_df):
    """Generate CSV export with metadata header and proper data types"""
    
    logger = setup_logging('csv_export')
    logger.info("Starting enhanced CSV export with data type validation")
    
    # Ensure proper data types before export
    export_df = filtered_df.copy()
    conversion_warnings = []
    
    try:
        # Convert categorical columns to numeric where appropriate
        export_df, conversion_reports = convert_categorical_to_numeric(export_df, logger=logger)
        
        # Track conversion warnings for metadata
        for col, report in conversion_reports.items():
            if report['conversion_successful']:
                if report['values_failed'] > 0:
                    conversion_warnings.append(f"Column '{col}': {report['values_failed']} values converted to null")
            else:
                conversion_warnings.append(f"Column '{col}': Conversion failed - exported as original type")
        
        logger.info(f"Data type validation complete: {len(conversion_reports)} columns processed")
        
    except Exception as e:
        logger.error(f"Error during data type conversion for export: {str(e)}")
        conversion_warnings.append(f"Data type conversion error: {str(e)}")
    
    # Create metadata header
    metadata_lines = [
        "# Property Filter Dashboard Export",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Original Data Source: {st.session_state.get('uploaded_filename', 'Unknown')}",
        f"# Upload Date: {st.session_state.get('upload_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if st.session_state.get('upload_timestamp') else 'Unknown'}",
        f"# Original Records: {len(original_df):,}",
        f"# Filtered Records: {len(filtered_df):,}",
        f"# Filter Pass Rate: {len(filtered_df)/len(original_df)*100:.1f}%",
        "#",
        "# Data Quality Metrics:",
    ]
    
    if st.session_state.get('validation_result'):
        validation_result = st.session_state['validation_result']
        metadata_lines.append(f"# Data Quality Score: {validation_result['data_quality_score']:.0f}/100")
        if validation_result['warnings']:
            metadata_lines.append(f"# Warnings: {len(validation_result['warnings'])}")
    
    # Add data type conversion information
    if conversion_warnings:
        metadata_lines.extend([
            "#",
            "# Data Type Conversions:"
        ])
        for warning in conversion_warnings:
            metadata_lines.append(f"# {warning}")
    
    # Add column type information
    metadata_lines.extend([
        "#",
        "# Column Data Types:"
    ])
    for col in export_df.columns:
        dtype_str = str(export_df[col].dtype)
        metadata_lines.append(f"# {col}: {dtype_str}")
    
    metadata_lines.extend([
        "#",
        "# Column Definitions:",
        "# Property Name: Unique property identifier",
        "# Property Type: Type of property (Industrial, Warehouse, etc.)",
        "# Building SqFt: Building square footage (numeric)",
        "# Lot Size Acres: Lot size in acres (numeric)",
        "# Sold Price: Property sale price (numeric)",
        "#",
        ""
    ])
    
    # Convert DataFrame to CSV with proper data types
    try:
        csv_data = export_df.to_csv(index=False)
        logger.info("CSV export successful")
    except Exception as e:
        logger.error(f"Error generating CSV: {str(e)}")
        # Fallback to original DataFrame
        csv_data = filtered_df.to_csv(index=False)
        metadata_lines.append(f"# Export Warning: Used original data types due to conversion error: {str(e)}")
    
    # Combine metadata and data
    full_csv = "\n".join(metadata_lines) + csv_data
    
    return full_csv

def generate_enhanced_excel_export(filtered_df, original_df):
    """Generate Excel export with metadata sheet and proper data types"""
    
    logger = setup_logging('excel_export')
    logger.info("Starting enhanced Excel export with data type validation")
    
    buffer = io.BytesIO()
    
    # Ensure proper data types before export
    export_df = filtered_df.copy()
    conversion_reports = {}
    
    try:
        # Convert categorical columns to numeric where appropriate
        export_df, conversion_reports = convert_categorical_to_numeric(export_df, logger=logger)
        logger.info(f"Data type validation complete: {len(conversion_reports)} columns processed")
        
    except Exception as e:
        logger.error(f"Error during data type conversion for export: {str(e)}")
        # Use original DataFrame if conversion fails
        export_df = filtered_df.copy()
    
    try:
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Write filtered data to main sheet with proper data types
            export_df.to_excel(writer, sheet_name='Filtered Properties', index=False)
            
            # Create metadata sheet
            metadata_data = {
                'Metric': [
                    'Export Generated',
                    'Original Data Source',
                    'Upload Date',
                    'Processing Time',
                    'Original Records',
                    'Filtered Records',
                    'Filter Pass Rate',
                    'Data Quality Score',
                    'Memory Usage',
                    'Columns Cleaned',
                    'Data Type Conversions'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    st.session_state.get('uploaded_filename', 'Unknown'),
                    st.session_state.get('upload_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if st.session_state.get('upload_timestamp') else 'Unknown',
                    f"{st.session_state.get('processing_time', 0):.2f} seconds",
                    f"{len(original_df):,}",
                    f"{len(filtered_df):,}",
                    f"{len(filtered_df)/len(original_df)*100:.1f}%",
                    f"{st.session_state.get('validation_result', {}).get('data_quality_score', 0):.0f}/100",
                    f"{st.session_state.get('processing_report', {}).get('memory_usage', 0):.1f} MB",
                    len(st.session_state.get('processing_report', {}).get('columns_cleaned', [])),
                    len(conversion_reports)
                ]
            }
            
            metadata_df = pd.DataFrame(metadata_data)
            metadata_df.to_excel(writer, sheet_name='Export Metadata', index=False)
            
            # Create data type information sheet
            if conversion_reports:
                dtype_data = []
                for col, report in conversion_reports.items():
                    dtype_data.append({
                        'Column': col,
                        'Original Type': report['original_dtype'],
                        'Final Type': report['target_dtype'],
                        'Conversion Success': 'Yes' if report['conversion_successful'] else 'No',
                        'Values Converted': report['values_converted'],
                        'Values Failed': report['values_failed'],
                        'Conversion Method': report['conversion_method'],
                        'Warnings': '; '.join(report['warnings']) if report['warnings'] else 'None'
                    })
                
                if dtype_data:
                    dtype_df = pd.DataFrame(dtype_data)
                    dtype_df.to_excel(writer, sheet_name='Data Type Conversions', index=False)
            
            # Add column data types sheet
            column_types_data = []
            for col in export_df.columns:
                column_types_data.append({
                    'Column': col,
                    'Data Type': str(export_df[col].dtype),
                    'Non-Null Count': export_df[col].count(),
                    'Null Count': export_df[col].isna().sum(),
                    'Sample Values': str(export_df[col].dropna().head(3).tolist()) if export_df[col].notna().any() else 'No data'
                })
            
            column_types_df = pd.DataFrame(column_types_data)
            column_types_df.to_excel(writer, sheet_name='Column Information', index=False)
            
            # Add processing report if available
            if st.session_state.get('processing_report'):
                processing_report = st.session_state['processing_report']
                
                # Create processing details sheet
                processing_data = []
                
                if processing_report.get('cleaning_stats'):
                    for col, actions in processing_report['cleaning_stats'].items():
                        for action in actions:
                            processing_data.append({
                                'Column': col,
                                'Action': action,
                                'Type': 'Data Cleaning'
                            })
                
                if processing_report.get('outliers_detected'):
                    for col, count in processing_report['outliers_detected'].items():
                        processing_data.append({
                            'Column': col,
                            'Action': f'{count} outliers detected',
                            'Type': 'Quality Check'
                        })
                
                # Add data type conversion actions
                for col, report in conversion_reports.items():
                    if report['conversion_successful']:
                        processing_data.append({
                            'Column': col,
                            'Action': f"Converted from {report['original_dtype']} to {report['target_dtype']}",
                            'Type': 'Data Type Conversion'
                        })
                
                if processing_data:
                    processing_df = pd.DataFrame(processing_data)
                    processing_df.to_excel(writer, sheet_name='Processing Details', index=False)
        
        logger.info("Excel export successful")
        
    except Exception as e:
        logger.error(f"Error generating Excel export: {str(e)}")
        # Fallback: create simple export with original data
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, sheet_name='Filtered Properties', index=False)
            
            # Add error information
            error_df = pd.DataFrame({
                'Export Error': [str(e)],
                'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Note': ['Export completed with original data types due to conversion error']
            })
            error_df.to_excel(writer, sheet_name='Export Warnings', index=False)
    
    return buffer

def generate_data_quality_report():
    """Generate comprehensive data quality report"""
    report_lines = [
        "PROPERTY DATA QUALITY REPORT",
        "=" * 50,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Data Source: {st.session_state.get('uploaded_filename', 'Unknown')}",
        f"Upload Date: {st.session_state.get('upload_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if st.session_state.get('upload_timestamp') else 'Unknown'}",
        ""
    ]
    
    # Add validation results
    if st.session_state.get('validation_result'):
        validation_result = st.session_state['validation_result']
        report_lines.extend([
            "DATA VALIDATION RESULTS:",
            "-" * 30,
            f"Overall Quality Score: {validation_result.get('data_quality_score', 0):.0f}/100",
            f"Total Rows: {validation_result.get('row_count', 0):,}",
            f"Total Columns: {validation_result.get('column_count', 0)}",
            f"Required Columns Found: {len(validation_result.get('required_columns_found', []))}",
            f"Required Columns Missing: {len(validation_result.get('required_columns_missing', []))}",
            ""
        ])
        
        if validation_result['warnings']:
            report_lines.extend([
                "WARNINGS:",
                "-" * 15
            ])
            for warning in validation_result['warnings']:
                report_lines.append(f"• {warning}")
            report_lines.append("")
    
    # Add processing report
    if st.session_state.get('processing_report'):
        processing_report = st.session_state['processing_report']
        report_lines.extend([
            "DATA PROCESSING SUMMARY:",
            "-" * 30,
            f"Processing Time: {processing_report.get('processing_time', 0):.2f} seconds",
            f"Memory Usage: {processing_report.get('memory_usage', 0):.1f} MB",
            f"Columns Cleaned: {len(processing_report.get('columns_cleaned', []))}",
            ""
        ])
        
        if processing_report.get('cleaning_stats'):
            report_lines.extend([
                "CLEANING ACTIONS PERFORMED:",
                "-" * 35
            ])
            for col, actions in processing_report['cleaning_stats'].items():
                report_lines.append(f"\n{col}:")
                for action in actions:
                    report_lines.append(f"  • {action}")
            report_lines.append("")
        
        if processing_report.get('recommendations'):
            report_lines.extend([
                "RECOMMENDATIONS:",
                "-" * 20
            ])
            for rec in processing_report['recommendations']:
                report_lines.append(f"• {rec}")
            report_lines.append("")
    
    return "\n".join(report_lines)

def fuzzy_match_columns(df_columns, target_columns):
    """Perform fuzzy matching of column names"""
    from difflib import SequenceMatcher
    
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    matches = {}
    confidence_scores = {}
    
    for target_col in target_columns:
        best_match = None
        best_score = 0
        
        for df_col in df_columns:
            score = similarity(target_col, df_col)
            if score > best_score and score > 0.6:  # Minimum 60% similarity
                best_score = score
                best_match = df_col
        
        if best_match:
            matches[best_match] = target_col
            confidence_scores[best_match] = best_score
    
    return matches, confidence_scores

def suggest_column_mapping(df):
    """Suggest column mappings for non-standard column names"""
    # Define standard column names and their common variations
    standard_columns = {
        'Property Name': ['property_name', 'prop_name', 'name', 'property', 'building_name'],
        'Property Type': ['property_type', 'prop_type', 'type', 'building_type', 'asset_type'],
        'Address': ['address', 'street_address', 'location', 'street'],
        'City': ['city', 'municipality', 'town'],
        'County': ['county', 'parish'],
        'State': ['state', 'province', 'region'],
        'Building SqFt': ['building_sqft', 'bldg_sqft', 'sqft', 'square_feet', 'building_sf', 'sf'],
        'Lot Size Acres': ['lot_size_acres', 'lot_acres', 'acres', 'lot_size', 'land_acres'],
        'Year Built': ['year_built', 'built_year', 'construction_year', 'year_constructed'],
        'Sold Price': ['sold_price', 'sale_price', 'price', 'purchase_price', 'value'],
        'Occupancy': ['occupancy', 'occupancy_rate', 'occupied', 'vacancy_rate']
    }
    
    mapping_result = {
        'exact_matches': {},
        'fuzzy_matches': {},
        'unmapped_required': [],
        'unmapped_recommended': [],
        'confidence_scores': {}
    }
    
    df_columns = df.columns.tolist()
    
    # Check for exact matches first
    for standard_col, variations in standard_columns.items():
        for df_col in df_columns:
            if df_col == standard_col:
                mapping_result['exact_matches'][df_col] = standard_col
                break
            elif df_col.lower() in [v.lower() for v in variations]:
                mapping_result['exact_matches'][df_col] = standard_col
                break
    
    # Find fuzzy matches for unmapped columns
    unmapped_df_cols = [col for col in df_columns if col not in mapping_result['exact_matches']]
    unmapped_standard_cols = [col for col in standard_columns.keys() if col not in mapping_result['exact_matches'].values()]
    
    if unmapped_df_cols and unmapped_standard_cols:
        fuzzy_matches, confidence_scores = fuzzy_match_columns(unmapped_df_cols, unmapped_standard_cols)
        mapping_result['fuzzy_matches'] = fuzzy_matches
        mapping_result['confidence_scores'] = confidence_scores
    
    # Identify unmapped required columns
    required_columns = ['Property Type', 'City', 'State']
    all_mapped = list(mapping_result['exact_matches'].values()) + list(mapping_result['fuzzy_matches'].values())
    
    for req_col in required_columns:
        if req_col not in all_mapped:
            mapping_result['unmapped_required'].append(req_col)
    
    return mapping_result

def validate_data_structure(df):
    """Validate the structure of uploaded data with enhanced column mapping"""
    validation_result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'required_columns_found': [],
        'required_columns_missing': [],
        'suggested_mappings': {},
        'data_quality_score': 0,
        'row_count': len(df),
        'column_count': len(df.columns),
        'column_mapping': None
    }
    
    # Get column mapping suggestions
    column_mapping = suggest_column_mapping(df)
    validation_result['column_mapping'] = column_mapping
    
    # Define required and optional columns
    required_columns = ['Property Type', 'City', 'State']
    recommended_columns = ['Property Name', 'Building SqFt', 'Lot Size Acres', 'County']
    optional_columns = ['Address', 'Year Built', 'Sold Price', 'Loan Amount', 'Interest Rate', 
                       'Number of Units', 'Lot Size SqFt', 'Occupancy']
    
    # Check for required columns (including mapped ones)
    all_mapped_columns = list(column_mapping['exact_matches'].values()) + list(column_mapping['fuzzy_matches'].values())
    
    for col in required_columns:
        if col in df.columns or col in all_mapped_columns:
            validation_result['required_columns_found'].append(col)
        else:
            validation_result['required_columns_missing'].append(col)
            validation_result['errors'].append(f"Required column '{col}' is missing")
    
    # Add suggested mappings to validation result
    if column_mapping['fuzzy_matches']:
        for original_col, suggested_col in column_mapping['fuzzy_matches'].items():
            confidence = column_mapping['confidence_scores'].get(original_col, 0)
            validation_result['suggested_mappings'][original_col] = {
                'suggested': suggested_col,
                'confidence': confidence
            }
    
    # Check for recommended columns
    missing_recommended = []
    for col in recommended_columns:
        if col not in df.columns and col not in all_mapped_columns:
            missing_recommended.append(col)
    
    if missing_recommended:
        validation_result['warnings'].append(f"Recommended columns missing: {', '.join(missing_recommended)}")
    
    # Enhanced data quality scoring
    total_possible_columns = len(required_columns) + len(recommended_columns) + len(optional_columns)
    found_columns = len([col for col in required_columns + recommended_columns + optional_columns 
                        if col in df.columns or col in all_mapped_columns])
    column_score = (found_columns / total_possible_columns) * 40
    
    # Data completeness score
    non_null_ratio = df.count().sum() / (len(df) * len(df.columns))
    completeness_score = non_null_ratio * 30
    
    # Data type appropriateness score
    numeric_cols_expected = ['Building SqFt', 'Lot Size Acres', 'Year Built', 'Sold Price', 'Occupancy']
    numeric_score = 0
    for col in numeric_cols_expected:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == 'object':
                numeric_score += 1
    type_score = (numeric_score / len(numeric_cols_expected)) * 30
    
    validation_result['data_quality_score'] = min(100, column_score + completeness_score + type_score)
    
    # Set overall success
    validation_result['success'] = len(validation_result['errors']) == 0
    
    return validation_result

def apply_column_mapping(df, mapping_dict):
    """Apply column mapping to DataFrame"""
    df_mapped = df.copy()
    
    # Rename columns based on mapping
    rename_dict = {}
    for original_col, target_col in mapping_dict.items():
        if original_col in df_mapped.columns:
            rename_dict[original_col] = target_col
    
    if rename_dict:
        df_mapped = df_mapped.rename(columns=rename_dict)
    
    return df_mapped, list(rename_dict.keys())

def render_data_preview(df, validation_result, processing_report):
    """Render comprehensive data preview and processing report"""
    st.markdown("---")
    st.subheader("📊 Data Preview & Analysis")
    
    # Create tabs for different views
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Data Preview", "Processing Report", "Column Analysis"])
    
    with preview_tab1:
        st.markdown("### 📋 First 20 Rows")
        
        # Show basic dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{processing_report['memory_usage']:.1f} MB")
        with col4:
            st.metric("Quality Score", f"{validation_result['data_quality_score']:.0f}/100")
        
        # Display first 20 rows
        preview_df = df.head(20)
        st.dataframe(preview_df, use_container_width=True, height=400)
        
        # Navigation options
        st.markdown("### 🚀 Next Steps")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("Click 'Proceed to Filtering Dashboard' below to continue to the main dashboard.")
        
        with col2:
            if st.button("📊 View Full Report", use_container_width=True):
                st.session_state.show_full_report = True
        
        with col3:
            if st.button("🔄 Upload Different File", use_container_width=True):
                # Clean up session data and memory
                cleanup_session_data()
                # Clear upload state
                for key in ['file_uploaded', 'uploaded_data', 'data_loaded', 'validation_result', 'processing_report', 'show_filters', 'proceed_to_dashboard']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    with preview_tab2:
        st.markdown("### Processing Summary")
        
        # Processing metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**File Processing:**")
            st.write(f"• Original shape: {processing_report['original_shape'][0]:,} rows × {processing_report['original_shape'][1]} columns")
            st.write(f"• Final shape: {processing_report['final_shape'][0]:,} rows × {processing_report['final_shape'][1]} columns")
            st.write(f"• Processing time: {processing_report['processing_time']:.2f} seconds")
            st.write(f"• Memory usage: {processing_report['memory_usage']:.1f} MB")
        
        with col2:
            st.markdown("**Data Cleaning:**")
            if processing_report['columns_cleaned']:
                st.write(f"• Columns cleaned: {len(processing_report['columns_cleaned'])}")
                for col in processing_report['columns_cleaned']:
                    st.write(f"  - {col}")
            else:
                st.write("• No columns required cleaning")
        
        # Null value handling
        if processing_report['null_handling']:
            st.markdown("**Null Values Handled:**")
            for col, count in processing_report['null_handling'].items():
                st.write(f"• {col}: {count:,} null values")
        
        # Recommendations
        if processing_report['recommendations']:
            st.markdown("**Recommendations:**")
            for rec in processing_report['recommendations']:
                st.info(f"💡 {rec}")
    
    with preview_tab3:
        st.markdown("### 🔍 Column Analysis")
        
        # Create column analysis
        column_analysis = []
        for col in df.columns:
            analysis = {
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': f"{df[col].count():,}",
                'Null Count': f"{df[col].isnull().sum():,}",
                'Unique Values': f"{df[col].nunique():,}",
                'Sample Values': ', '.join([str(x) for x in df[col].dropna().head(3).tolist()])
            }
            column_analysis.append(analysis)
        
        analysis_df = pd.DataFrame(column_analysis)
        st.dataframe(analysis_df, use_container_width=True, height=400)
        
        # Data quality indicators
        st.markdown("### 🎯 Data Quality Indicators")
        
        # Calculate quality metrics
        total_cells = len(df) * len(df.columns)
        non_null_cells = df.count().sum()
        completeness = (non_null_cells / total_cells) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Completeness", f"{completeness:.1f}%")
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        with col3:
            text_cols = df.select_dtypes(include=['object']).columns
            st.metric("Text Columns", len(text_cols))
        
        # Show data quality issues if any
        quality_issues = []
        
        # Check for columns with high null percentage
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 50:
                quality_issues.append(f"Column '{col}' has {null_pct:.1f}% missing values")
        
        # Check for columns with very low uniqueness (potential data quality issues)
        for col in df.select_dtypes(include=['object']).columns:
            unique_pct = (df[col].nunique() / len(df)) * 100
            if unique_pct < 5 and df[col].nunique() > 1:
                quality_issues.append(f"Column '{col}' has low diversity ({unique_pct:.1f}% unique values)")
        
        if quality_issues:
            st.markdown("### ⚠️ Potential Data Quality Issues")
            for issue in quality_issues:
                st.warning(issue)

def detect_outliers(df, column):
    """Detect outliers in numeric columns using IQR method"""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return []
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers.index.tolist()

def advanced_data_cleaning(df):
    """Perform advanced data cleaning with detailed tracking"""
    cleaning_report = {
        'columns_processed': [],
        'cleaning_actions': {},
        'outliers_detected': {},
        'data_transformations': {},
        'quality_improvements': {}
    }
    
    df_cleaned = df.copy()
    
    # Define numeric columns for cleaning
    numeric_columns = [
        'Building SqFt', 'Sold Price', 'Lot Size Acres', 
        'Year Built', 'Loan Amount', 'Interest Rate',
        'Number of Units', 'Lot Size SqFt', 'Occupancy'
    ]
    
    for col in numeric_columns:
        if col in df_cleaned.columns:
            cleaning_report['columns_processed'].append(col)
            actions = []
            
            # Store original state
            original_nulls = df_cleaned[col].isnull().sum()
            original_dtype = df_cleaned[col].dtype
            
            # Clean the column
            try:
                df_cleaned[col] = clean_numeric_column(df_cleaned[col])
                actions.append("Applied numeric cleaning (removed currency symbols, commas, percentages)")
                
                # Check for improvements
                new_nulls = df_cleaned[col].isnull().sum()
                new_dtype = df_cleaned[col].dtype
                
                if new_dtype != original_dtype:
                    cleaning_report['data_transformations'][col] = f"{original_dtype} → {new_dtype}"
                    actions.append(f"Converted data type from {original_dtype} to {new_dtype}")
                
                if new_nulls != original_nulls:
                    null_change = original_nulls - new_nulls
                    if null_change > 0:
                        cleaning_report['quality_improvements'][col] = f"Recovered {null_change} values from text formatting"
                        actions.append(f"Recovered {null_change} values from text formatting")
                
                # Detect outliers
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    outlier_indices = detect_outliers(df_cleaned, col)
                    if outlier_indices:
                        cleaning_report['outliers_detected'][col] = len(outlier_indices)
                        actions.append(f"Detected {len(outlier_indices)} potential outliers")
                
            except Exception as e:
                actions.append(f"Cleaning failed: {str(e)}")
            
            cleaning_report['cleaning_actions'][col] = actions
    
    return df_cleaned, cleaning_report

def calculate_data_quality_score(df, validation_result):
    """Calculate comprehensive data quality score"""
    scores = {
        'completeness': 0,
        'consistency': 0,
        'validity': 0,
        'accuracy': 0
    }
    
    # Completeness Score (40% weight)
    total_cells = len(df) * len(df.columns)
    non_null_cells = df.count().sum()
    scores['completeness'] = (non_null_cells / total_cells) * 40
    
    # Consistency Score (25% weight) - based on data types and formats
    consistency_score = 0
    numeric_cols = ['Building SqFt', 'Sold Price', 'Lot Size Acres', 'Year Built', 'Occupancy']
    for col in numeric_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                consistency_score += 1
    
    if numeric_cols:
        scores['consistency'] = (consistency_score / len(numeric_cols)) * 25
    
    # Validity Score (20% weight) - based on required columns presence
    required_cols = ['Property Type', 'City', 'State']
    validity_score = sum(1 for col in required_cols if col in df.columns)
    scores['validity'] = (validity_score / len(required_cols)) * 20
    
    # Accuracy Score (15% weight) - based on reasonable value ranges
    accuracy_score = 0
    accuracy_checks = 0
    
    if 'Year Built' in df.columns:
        accuracy_checks += 1
        valid_years = df['Year Built'][(df['Year Built'] >= 1800) & (df['Year Built'] <= 2025)]
        if len(df['Year Built'].dropna()) > 0:
            accuracy_score += len(valid_years) / len(df['Year Built'].dropna())
    
    if 'Building SqFt' in df.columns:
        accuracy_checks += 1
        valid_sqft = df['Building SqFt'][(df['Building SqFt'] > 0) & (df['Building SqFt'] < 10000000)]
        if len(df['Building SqFt'].dropna()) > 0:
            accuracy_score += len(valid_sqft) / len(df['Building SqFt'].dropna())
    
    if accuracy_checks > 0:
        scores['accuracy'] = (accuracy_score / accuracy_checks) * 15
    
    total_score = sum(scores.values())
    return total_score, scores

def generate_processing_report(df, original_df, processing_time):
    """Generate a comprehensive processing report with advanced analytics"""
    report = {
        'original_shape': original_df.shape if original_df is not None else df.shape,
        'final_shape': df.shape,
        'columns_cleaned': [],
        'cleaning_stats': {},
        'null_handling': {},
        'type_conversions': {},
        'outliers_detected': {},
        'processing_time': processing_time,
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        'recommendations': [],
        'quality_breakdown': {}
    }
    
    # Perform advanced cleaning analysis
    df_for_analysis, cleaning_report = advanced_data_cleaning(df)
    
    # Merge cleaning report into main report
    report['columns_cleaned'] = cleaning_report['columns_processed']
    report['cleaning_stats'] = cleaning_report['cleaning_actions']
    report['outliers_detected'] = cleaning_report['outliers_detected']
    report['type_conversions'] = cleaning_report['data_transformations']
    
    # Calculate null handling
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            report['null_handling'][col] = null_count
    
    # Calculate quality score breakdown
    validation_result = validate_data_structure(df)
    quality_score, quality_breakdown = calculate_data_quality_score(df, validation_result)
    report['quality_breakdown'] = quality_breakdown
    
    # Generate enhanced recommendations
    recommendations = []
    
    if report['memory_usage'] > 100:
        recommendations.append("Large dataset detected (>100MB). Consider filtering data before analysis for better performance.")
    
    if len(report['null_handling']) > 0:
        high_null_cols = [col for col, count in report['null_handling'].items() if count / len(df) > 0.3]
        if high_null_cols:
            recommendations.append(f"High missing data in columns: {', '.join(high_null_cols)}. Consider data imputation or removal.")
    
    if quality_score < 70:
        recommendations.append("Data quality score is below 70%. Review column structure and data completeness.")
    
    if report['outliers_detected']:
        total_outliers = sum(report['outliers_detected'].values())
        recommendations.append(f"Detected {total_outliers} potential outliers across numeric columns. Review for data accuracy.")
    
    if quality_breakdown['completeness'] < 30:
        recommendations.append("Low data completeness detected. Many cells are empty - consider data validation at source.")
    
    if quality_breakdown['consistency'] < 15:
        recommendations.append("Data type inconsistencies found. Some numeric columns may contain text values.")
    
    report['recommendations'] = recommendations
    
    return report

def secure_file_validation(uploaded_file_content, filename):
    """Perform security validation on uploaded file content"""
    try:
        # Check for suspicious file patterns
        if len(uploaded_file_content) < 100:  # Too small to be a valid Excel file
            return False, "File appears to be too small to contain valid Excel data"
        
        # Basic Excel file signature check
        excel_signatures = [
            b'PK\x03\x04',  # XLSX signature (ZIP-based)
            b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',  # XLS signature (OLE2-based)
        ]
        
        file_start = uploaded_file_content[:8]
        is_excel = any(file_start.startswith(sig) for sig in excel_signatures)
        
        if not is_excel:
            return False, "File does not appear to be a valid Excel file based on content analysis"
        
        # Check for reasonable file structure
        try:
            file_buffer = io.BytesIO(uploaded_file_content)
            # Try to read just the structure without loading all data
            pd.read_excel(file_buffer, nrows=0)  # Just read headers
        except Exception:
            return False, "File structure appears to be invalid or corrupted"
        
        return True, "File passed security validation"
    
    except Exception as e:
        return False, f"Security validation failed: {str(e)}"

@st.cache_data(ttl=3600, max_entries=3)  # Cache for 1 hour, max 3 files
def load_uploaded_data(uploaded_file_content, filename):
    """Load and process uploaded Excel file with security and performance optimizations"""
    import time
    start_time = time.time()
    
    # Timeout protection
    PROCESSING_TIMEOUT = 300  # 5 minutes
    
    try:
        # Security validation
        is_secure, security_message = secure_file_validation(uploaded_file_content, filename)
        if not is_secure:
            return None, None, f"Security check failed: {security_message}"
        
        # Create a BytesIO object from the uploaded file content
        file_buffer = io.BytesIO(uploaded_file_content)
        
        # Memory-efficient loading for large files
        try:
            # First, try to load with chunking for very large files
            original_df = pd.read_excel(file_buffer, engine='openpyxl')
        except MemoryError:
            # If memory error, try with different engine or chunking
            file_buffer.seek(0)  # Reset buffer position
            try:
                original_df = pd.read_excel(file_buffer, engine='xlrd')
            except:
                return None, None, "File is too large to process. Please reduce file size or split into smaller files."
        
        # Check processing time
        if time.time() - start_time > PROCESSING_TIMEOUT:
            return None, None, "File processing timed out. Please try with a smaller file."
        
        if original_df.empty:
            return None, None, "The uploaded file contains no data"
        
        # Limit maximum rows for performance
        MAX_ROWS = 100000  # 100k rows maximum
        if len(original_df) > MAX_ROWS:
            return None, None, f"File contains {len(original_df):,} rows. Maximum allowed is {MAX_ROWS:,} rows. Please filter your data or split into smaller files."
        
        # Create a copy for processing
        df = original_df.copy()
        
        # Optimize memory usage
        df = optimize_dataframe_memory(df)
        
        # Validate data structure
        validation_result = validate_data_structure(df)
        
        if not validation_result['success']:
            error_msg = "Data validation failed:\n" + "\n".join(validation_result['errors'])
            return None, validation_result, error_msg
        
        # Clean numeric columns with error handling and timeout protection
        numeric_columns = [
            'Building SqFt', 'Sold Price', 'Lot Size Acres', 
            'Year Built', 'Loan Amount', 'Interest Rate',
            'Number of Units', 'Lot Size SqFt', 'Occupancy'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Check timeout
                    if time.time() - start_time > PROCESSING_TIMEOUT:
                        return None, None, "File processing timed out during data cleaning."
                    
                    df[col] = clean_numeric_column(df[col])
                except Exception as clean_error:
                    validation_result['warnings'].append(f"Could not clean column '{col}': {str(clean_error)}")
                    continue
        
        return df, validation_result, "File loaded and validated successfully"
    
    except MemoryError:
        return None, None, "Insufficient memory to process this file. Please reduce file size or try with a smaller dataset."
    except Exception as e:
        return None, None, f"Error loading file: {str(e)}"

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage"""
    # Convert object columns to category where appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            try:
                df[col] = df[col].astype('category')
            except:
                pass  # Keep as object if conversion fails
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        try:
            # Handle potential categorical data
            if df[col].dtype.name == 'category':
                series = pd.to_numeric(df[col].astype(str), errors='coerce')
            else:
                series = pd.to_numeric(df[col], errors='coerce')

            series = series.dropna()
            if len(series) > 0:
                col_min = series.min()
                col_max = series.max()
            else:
                continue
        except:
            continue  # Skip if unable to process
        
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    return df

def cleanup_session_data():
    """Clean up session data to free memory"""
    # List of keys to clean up when uploading new file
    cleanup_keys = [
        'uploaded_data', 'validation_result', 'processing_report',
        'filtered_df', 'filter_params'
    ]
    
    for key in cleanup_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Force garbage collection
    import gc
    gc.collect()

def render_upload_interface():
    """Render the enhanced file upload interface with drag-and-drop and guidance"""
    st.subheader("📁 Upload Your Property Data")
    
    # Create upload area with custom styling
    st.markdown("""
    <style>
    .upload-area {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
        margin: 10px 0;
    }
    .upload-area:hover {
        border-color: #007bff;
        background-color: #e3f2fd;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Upload instructions
    st.markdown("""
    <div class="upload-area">
        <h4>📤 Drag and drop your Excel file here</h4>
        <p>or click below to browse your files</p>
        <p><small>Supported formats: .xlsx, .xls | Maximum size: 50MB</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with enhanced options
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload your property data in Excel format (.xlsx or .xls). Maximum file size: 50MB",
        label_visibility="collapsed"
    )
    
    # Show upload progress if file is being processed
    if st.session_state.upload_status == 'processing':
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f'Processing file... {i + 1}%')
            # In a real implementation, this would be tied to actual processing progress
    
    if uploaded_file is not None:
        # Validate the uploaded file
        is_valid, message, error_type = validate_uploaded_file(uploaded_file)
        
        if is_valid:
            # Update session state
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.file_size = uploaded_file.size
            st.session_state.upload_timestamp = datetime.now()
            st.session_state.upload_status = 'processing'
            
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"File: {uploaded_file.name}")
            with col2:
                st.info(f"Size: {uploaded_file.size / (1024*1024):.1f} MB")
            with col3:
                st.info(f"Type: .{uploaded_file.name.split('.')[-1]}")
            
            # Process the file
            with st.spinner("Processing your file..."):
                start_time = datetime.now()
                
                # Load the data with validation
                df, validation_result, load_message = load_uploaded_data(uploaded_file.getvalue(), uploaded_file.name)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                st.session_state.processing_time = processing_time
                
                if df is not None and validation_result is not None:
                    # Generate processing report
                    processing_report = generate_processing_report(df, None, processing_time)
                    
                    # Success
                    st.session_state.uploaded_data = df
                    st.session_state.validation_result = validation_result
                    st.session_state.processing_report = processing_report
                    st.session_state.data_loaded = True
                    st.session_state.upload_status = 'complete'
                    st.session_state.validation_passed = validation_result['success']
                    st.session_state.file_uploaded = True
                    
                    st.success(f"✅ {load_message}")
                    
                    # Show validation results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Data Quality Score", f"{validation_result['data_quality_score']:.0f}/100")
                    with col2:
                        st.metric("Rows Processed", f"{len(df):,}")
                    with col3:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    
                    # Show column mapping suggestions if any
                    if validation_result['suggested_mappings']:
                        with st.expander("🔄 Column Mapping Suggestions"):
                            st.info("We found some columns that might match standard names:")
                            
                            mapping_to_apply = {}
                            for original_col, mapping_info in validation_result['suggested_mappings'].items():
                                suggested_col = mapping_info['suggested']
                                confidence = mapping_info['confidence']
                                
                                col1, col2, col3 = st.columns([2, 1, 2])
                                with col1:
                                    st.write(f"**{original_col}**")
                                with col2:
                                    st.write("→")
                                with col3:
                                    st.write(f"**{suggested_col}** ({confidence:.0%} match)")
                                
                                # Add checkbox to accept mapping
                                if st.checkbox(f"Apply mapping for '{original_col}'", key=f"map_{original_col}"):
                                    mapping_to_apply[original_col] = suggested_col
                            
                            # Apply mappings if any selected
                            if mapping_to_apply and st.button("Apply Selected Mappings"):
                                df_mapped, mapped_columns = apply_column_mapping(df, mapping_to_apply)
                                st.session_state.uploaded_data = df_mapped
                                st.success(f"Applied mappings for: {', '.join(mapped_columns)}")
                                st.rerun()
                            
                            # Show column mapping help
                            with st.expander("❓ Column Mapping Help"):
                                st.markdown("""
                                **What is Column Mapping?**
                                Column mapping helps match your column names to our standard format.
                                
                                **How it works:**
                                • We analyze your column names
                                • Suggest matches based on similarity
                                • You can accept or reject suggestions
                                
                                **Common Mappings:**
                                • "Bldg SqFt" → "Building SqFt"
                                • "Sale Price" → "Sold Price"  
                                • "Prop Type" → "Property Type"
                                • "Sq Ft" → "Building SqFt"
                                
                                **Tips:**
                                • Higher confidence scores are more reliable
                                • Review suggestions before applying
                                • You can always upload a new file if needed
                                """)
                    
                    # Show warnings if any
                    if validation_result['warnings']:
                        with st.expander("⚠️ Validation Warnings"):
                            for warning in validation_result['warnings']:
                                st.warning(warning)
                    
                    # Show data preview and processing report
                    render_data_preview(df, validation_result, processing_report)
                    
                    return df
                else:
                    # Error
                    st.session_state.upload_status = 'error'
                    st.session_state.validation_passed = False
                    st.error(f"❌ {load_message}")
                    
                    # Show validation errors if available
                    if validation_result and validation_result['errors']:
                        with st.expander("❌ Validation Errors"):
                            for error in validation_result['errors']:
                                st.error(error)
                    
                    return None
        else:
            # Invalid file - show detailed error guidance
            st.session_state.upload_status = 'error'
            show_error_guidance(error_type, message)
            return None
    
    else:
        # Show upload instructions
        st.info("👆 Please upload an Excel file to get started")
        
        # Enhanced help section
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📋 File Requirements & Format"):
                st.markdown("""
                **Supported Formats:**
                - Excel files (.xlsx, .xls)
                
                **File Size:**
                - Maximum 50MB
                
                **Required Columns:**
                - Property Type (e.g., "Industrial", "Warehouse")
                - City (property location)
                - State (property state)
                
                **Recommended Columns:**
                - Property Name (unique identifier)
                - Building SqFt (building square footage)
                - Lot Size Acres (lot size in acres)
                - County (property county)
                
                **Optional Columns:**
                - Address, Year Built, Sold Price
                - Loan Amount, Interest Rate
                - Number of Units, Occupancy
                """)
        
        with col2:
            with st.expander("💡 Tips for Best Results"):
                st.markdown("""
                **Data Formatting Tips:**
                - Use consistent column names
                - Remove special characters from numbers
                - Use standard date formats (YYYY-MM-DD)
                - Avoid merged cells in headers
                
                **Common Issues to Avoid:**
                - Empty rows at the top
                - Multiple header rows
                - Mixed data types in columns
                - Very large file sizes
                
                **Sample Data Format:**
                ```
                Property Name | Property Type | City | State
                Building A    | Industrial    | LA   | CA
                Warehouse B   | Distribution  | TX   | TX
                ```
                """)
        
        # Sample template download
        st.markdown("---")
        st.subheader("📥 Download Sample Template")
        
        # Create comprehensive sample data with multiple examples
        sample_data = {
            'Property Name': [
                'Industrial Park East Building A', 'Metro Distribution Center', 'Flex Manufacturing Hub',
                'Warehouse Complex North', 'Light Industrial Facility', 'Storage & Logistics Center',
                'Multi-Tenant Industrial', 'Cold Storage Warehouse', 'Manufacturing Plant Delta',
                'Distribution Hub Central'
            ],
            'Property Type': [
                'Industrial Warehouse', 'Distribution Center', 'Flex Space',
                'Warehouse', 'Light Industrial', 'Storage Facility',
                'Multi-Tenant Industrial', 'Cold Storage', 'Manufacturing',
                'Distribution Hub'
            ],
            'Address': [
                '123 Industrial Blvd', '456 Distribution Way', '789 Flex Street',
                '321 Warehouse Ave', '654 Industrial Pkwy', '987 Storage Lane',
                '147 Multi-Tenant Dr', '258 Cold Storage Rd', '369 Manufacturing St',
                '741 Distribution Cir'
            ],
            'City': [
                'Los Angeles', 'Houston', 'Phoenix', 'Dallas', 'Atlanta',
                'Chicago', 'Miami', 'Denver', 'Seattle', 'Las Vegas'
            ],
            'County': [
                'Los Angeles County', 'Harris County', 'Maricopa County', 'Dallas County', 'Fulton County',
                'Cook County', 'Miami-Dade County', 'Denver County', 'King County', 'Clark County'
            ],
            'State': ['CA', 'TX', 'AZ', 'TX', 'GA', 'IL', 'FL', 'CO', 'WA', 'NV'],
            'Building SqFt': [50000, 75000, 25000, 100000, 35000, 60000, 45000, 80000, 120000, 90000],
            'Lot Size Acres': [2.5, 5.0, 1.2, 8.0, 1.8, 3.5, 2.2, 4.5, 10.0, 6.5],
            'Year Built': [2000, 1995, 2010, 1985, 2005, 1998, 2012, 1990, 2008, 2015],
            'Sold Price': [1500000, 2250000, 800000, 3200000, 1100000, 1800000, 1350000, 2400000, 3800000, 2700000],
            'Loan Amount': [1200000, 1800000, 640000, 2560000, 880000, 1440000, 1080000, 1920000, 3040000, 2160000],
            'Interest Rate': [4.5, 3.8, 5.2, 4.1, 4.8, 3.9, 4.3, 4.6, 3.7, 4.0],
            'Number of Units': [1, 1, 3, 1, 2, 1, 5, 1, 1, 1],
            'Occupancy': [85, 90, 75, 95, 80, 88, 92, 100, 78, 85]
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Create Excel buffer for download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            sample_df.to_excel(writer, sheet_name='Property Data Template', index=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Excel Template",
                data=buffer.getvalue(),
                file_name="property_data_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download a sample Excel template with the correct column structure",
                use_container_width=True
            )
        
        with col2:
            # Create minimal template with just required columns
            minimal_data = {
                'Property Type': ['Industrial Warehouse', 'Distribution Center', 'Flex Space'],
                'City': ['Los Angeles', 'Houston', 'Phoenix'],
                'State': ['CA', 'TX', 'AZ']
            }
            minimal_df = pd.DataFrame(minimal_data)
            
            minimal_buffer = io.BytesIO()
            with pd.ExcelWriter(minimal_buffer, engine='openpyxl') as writer:
                minimal_df.to_excel(writer, sheet_name='Minimal Template', index=False)
            
            st.download_button(
                label="📥 Download Minimal Template",
                data=minimal_buffer.getvalue(),
                file_name="minimal_property_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download a minimal template with only required columns",
                use_container_width=True
            )
        
        # Show upload help
        show_upload_help()
        
        return None

@st.cache_data
def get_column_stats(df, column):
    """Cache column statistics for performance"""
    if column in df.columns and not df[column].isna().all():
        # Handle categorical data by converting to numeric first
        if df[column].dtype.name == 'category':
            series = pd.to_numeric(df[column].astype(str), errors='coerce')
        else:
            series = pd.to_numeric(df[column], errors='coerce')

        # Remove any NaN values for calculations
        series = series.dropna()

        if len(series) > 0:
            return {
                'min': series.min(),
                'max': series.max(),
                'mean': series.mean(),
                'count': series.count()
            }
    return None

@st.cache_data
def get_unique_values(df, column):
    """Cache unique values for dropdown filters"""
    if column in df.columns:
        return sorted(df[column].dropna().unique().tolist())
    return []

def validate_filter_ranges(size_range, lot_range, price_range, year_range, occupancy_range):
    """Validate that filter ranges are valid (min <= max)"""
    ranges = {
        'Building Size': size_range,
        'Lot Size': lot_range,
        'Price': price_range,
        'Year Built': year_range,
        'Occupancy': occupancy_range
    }
    
    for name, (min_val, max_val) in ranges.items():
        if min_val > max_val:
            st.error(f"Invalid {name} range: minimum ({min_val}) cannot be greater than maximum ({max_val})")
            return False
    return True

def apply_filters(df, filter_params):
    """Apply all active filters to the dataset with comprehensive error handling"""
    try:
        filtered_df = df.copy()
        
        # Property type filter
        if filter_params.get('industrial_keywords') and 'Property Type' in filtered_df.columns:
            try:
                pattern = '|'.join(filter_params['industrial_keywords'])
                type_mask = filtered_df['Property Type'].str.lower().str.contains(pattern, na=False)
                filtered_df = filtered_df[type_mask]
            except Exception as e:
                st.warning(f"Error applying property type filter: {str(e)}")
        
        # Building size filter
        if 'size_range' in filter_params and 'Building SqFt' in filtered_df.columns:
            try:
                from utils.data_type_utils import safe_numerical_comparison
                size_min_mask = safe_numerical_comparison(filtered_df['Building SqFt'], '>=', filter_params['size_range'][0], 'Building SqFt')
                size_max_mask = safe_numerical_comparison(filtered_df['Building SqFt'], '<=', filter_params['size_range'][1], 'Building SqFt')
                size_mask = size_min_mask & size_max_mask
                filtered_df = filtered_df[size_mask]
            except Exception as e:
                st.warning(f"Error applying building size filter: {str(e)}")
        
        # Lot size filter
        if 'lot_range' in filter_params and 'Lot Size Acres' in filtered_df.columns:
            try:
                lot_min_mask = safe_numerical_comparison(filtered_df['Lot Size Acres'], '>=', filter_params['lot_range'][0], 'Lot Size Acres')
                lot_max_mask = safe_numerical_comparison(filtered_df['Lot Size Acres'], '<=', filter_params['lot_range'][1], 'Lot Size Acres')
                lot_mask = lot_min_mask & lot_max_mask
                filtered_df = filtered_df[lot_mask]
            except Exception as e:
                st.warning(f"Error applying lot size filter: {str(e)}")
        
        # Price filter
        if filter_params.get('use_price_filter') and 'price_range' in filter_params:
            if 'Sold Price' in filtered_df.columns:
                try:
                    from utils.data_type_utils import safe_numerical_comparison
                    price_min_mask = safe_numerical_comparison(filtered_df['Sold Price'], '>=', filter_params['price_range'][0], 'Sold Price')
                    price_max_mask = safe_numerical_comparison(filtered_df['Sold Price'], '<=', filter_params['price_range'][1], 'Sold Price')
                    price_mask = price_min_mask & price_max_mask
                    filtered_df = filtered_df[price_mask]
                except Exception as e:
                    st.warning(f"Error applying price filter: {str(e)}")
        
        # Year built filter
        if filter_params.get('use_year_filter') and 'year_range' in filter_params:
            if 'Year Built' in filtered_df.columns:
                try:
                    year_min_mask = safe_numerical_comparison(filtered_df['Year Built'], '>=', filter_params['year_range'][0], 'Year Built')
                    year_max_mask = safe_numerical_comparison(filtered_df['Year Built'], '<=', filter_params['year_range'][1], 'Year Built')
                    year_mask = year_min_mask & year_max_mask
                    filtered_df = filtered_df[year_mask]
                except Exception as e:
                    st.warning(f"Error applying year built filter: {str(e)}")
        
        # Occupancy filter
        if filter_params.get('use_occupancy_filter') and 'occupancy_range' in filter_params:
            if 'Occupancy' in filtered_df.columns:
                try:
                    occupancy_min_mask = safe_numerical_comparison(filtered_df['Occupancy'], '>=', filter_params['occupancy_range'][0], 'Occupancy')
                    occupancy_max_mask = safe_numerical_comparison(filtered_df['Occupancy'], '<=', filter_params['occupancy_range'][1], 'Occupancy')
                    occupancy_mask = occupancy_min_mask & occupancy_max_mask
                    filtered_df = filtered_df[occupancy_mask]
                except Exception as e:
                    st.warning(f"Error applying occupancy filter: {str(e)}")
        
        # County and State filters
        if 'selected_counties' in filter_params and 'County' in filtered_df.columns:
            try:
                if filter_params['selected_counties']:  # Only apply if counties are selected
                    filtered_df = filtered_df[filtered_df['County'].isin(filter_params['selected_counties'])]
            except Exception as e:
                st.warning(f"Error applying county filter: {str(e)}")
        
        if 'selected_states' in filter_params and 'State' in filtered_df.columns:
            try:
                if filter_params['selected_states']:  # Only apply if states are selected
                    filtered_df = filtered_df[filtered_df['State'].isin(filter_params['selected_states'])]
            except Exception as e:
                st.warning(f"Error applying state filter: {str(e)}")
        
        return filtered_df
    
    except Exception as e:
        st.error(f"Critical error in filter application: {str(e)}")
        return df  # Return original dataframe if filtering fails completely

def main():
    st.title("Flex Property Filter Dashboard")
    st.markdown("### Interactive Property Filtering System")
    
    # Initialize states
    initialize_filter_state()
    initialize_upload_state()
    
    # Check if we have uploaded data or should use hardcoded file
    df = None
    
    # Try to use uploaded data first
    if st.session_state.data_loaded and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        st.success(f"Using uploaded file: {st.session_state.uploaded_filename}")
    else:
        # Show upload interface
        df = render_upload_interface()
        
        # If no uploaded data, try fallback to hardcoded file
        if df is None:
            st.markdown("---")
            st.subheader("Or use default data file")
            
            if st.button("Load Default Data File", help="Load data from the default file path"):
                file_path = r'C:\flex-property-pipeline\data\raw\Full Property Export.xlsx'
                with st.spinner("Loading default property data..."):
                    df = load_data(file_path)
                
                if not df.empty:
                    st.session_state.uploaded_data = df
                    st.session_state.data_loaded = True
                    st.session_state.uploaded_filename = "Full Property Export.xlsx (default)"
                    st.success("Default data loaded successfully!")
    
    # If still no data, stop here
    if df is None or df.empty:
        st.info("Please upload a file or load the default data to continue.")
        return
    
    # Convert categorical data to numeric to prevent runtime errors
    try:
        from utils.data_type_utils import convert_categorical_to_numeric
        df, conversion_reports = convert_categorical_to_numeric(df)
        
        # Show conversion info if any conversions were made
        if conversion_reports:
            converted_cols = [col for col, report in conversion_reports.items() if report['conversion_successful']]
            if converted_cols:
                st.info(f"✓ Converted {len(converted_cols)} categorical columns to numeric: {', '.join(converted_cols)}")
    except Exception as e:
        st.warning(f"Note: Some data type conversions may not be optimal: {str(e)}")
    
    # Handle transition from upload to filtering
    if st.session_state.get('show_filters', False):
        # Show loading transition
        with st.spinner("🚀 Transitioning to filtering dashboard..."):
            import time
            time.sleep(1)  # Brief pause for user feedback
            st.session_state.proceed_to_dashboard = True
            del st.session_state['show_filters']
            st.success("Ready for filtering!")
            st.rerun()
    
    # Check if user should see upload interface or filtering dashboard
    should_show_upload = (st.session_state.get('data_loaded', False) and 
                         st.session_state.get('upload_status') == 'complete' and 
                         not st.session_state.get('proceed_to_dashboard', False))
    
    # Debug information (can be removed in production)
    with st.sidebar.expander("Debug Info"):
        st.write(f"data_loaded: {st.session_state.get('data_loaded', False)}")
        st.write(f"upload_status: {st.session_state.get('upload_status', 'none')}")
        st.write(f"proceed_to_dashboard: {st.session_state.get('proceed_to_dashboard', False)}")
        st.write(f"should_show_upload: {should_show_upload}")
    
    if should_show_upload:
        # Stay on upload/preview interface but provide proceed button
        st.info("Data uploaded successfully! Ready to proceed to filtering.")
        
        # Add proceed button here since user can't access the data preview
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Proceed to Filtering Dashboard", type="primary", use_container_width=True):
                with st.spinner("Transitioning to filtering dashboard..."):
                    import time
                    time.sleep(1)  # Brief pause for user feedback
                    st.session_state.proceed_to_dashboard = True
                    st.success("Ready for filtering!")
                    st.rerun()
        
        # Show basic data info
        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Properties", f"{len(df):,}")
        with col2:
            if 'Property Type' in df.columns:
                industrial_count = df['Property Type'].str.lower().str.contains(
                    'industrial|warehouse|distribution|flex', na=False
                ).sum()
                st.metric("Industrial Properties", f"{industrial_count:,}")
            else:
                st.metric("Industrial Properties", "N/A")
        with col3:
            if 'Building SqFt' in df.columns:
                avg_sqft = df['Building SqFt'].mean()
                if not pd.isna(avg_sqft):
                    st.metric("Avg Building Size", f"{avg_sqft:,.0f} sqft")
                else:
                    st.metric("Avg Building Size", "N/A")
            else:
                st.metric("Avg Building Size", "N/A")
        with col4:
            unique_cities = df['City'].nunique() if 'City' in df.columns else 0
            st.metric("Unique Cities", unique_cities)
        
        # Show data preview
        st.subheader("Data Preview")
        preview_cols = ['Property Name', 'Property Type', 'City', 'State', 'Building SqFt'] if all(col in df.columns for col in ['Property Name', 'Property Type', 'City', 'State', 'Building SqFt']) else df.columns.tolist()[:5]
        st.dataframe(df[preview_cols].head(10), use_container_width=True)
        st.caption("Showing first 10 properties")
        
        return
    
    # Show data source information and current state
    if st.session_state.get('file_uploaded', False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📁 Data Source: {st.session_state.uploaded_filename}")
            # Debug info (remove in production)
            if st.session_state.get('proceed_to_dashboard', False):
                st.success("🎯 Filtering Dashboard Active")
            else:
                st.warning("⏳ Upload Mode - Click 'Proceed to Filtering' to continue")
        with col2:
            if st.button("🔄 Upload New File"):
                # Clean up session data and memory
                cleanup_session_data()
                # Clear upload state to return to upload interface
                for key in ['file_uploaded', 'uploaded_data', 'data_loaded', 'validation_result', 'processing_report', 'show_filters', 'proceed_to_dashboard']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Display data overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        if 'Property Type' in df.columns:
            industrial_count = df['Property Type'].str.lower().str.contains(
                'industrial|warehouse|distribution|flex', na=False
            ).sum()
            st.metric("Industrial Properties", f"{industrial_count:,}")
        else:
            st.metric("Industrial Properties", "N/A")
    with col3:
        if 'Building SqFt' in df.columns:
            from utils.data_type_utils import safe_mean_calculation
            avg_sqft = safe_mean_calculation(df['Building SqFt'], 'Building SqFt')
            if not pd.isna(avg_sqft):
                st.metric("Avg Building Size", f"{avg_sqft:,.0f} sqft")
            else:
                st.metric("Avg Building Size", "N/A")
        else:
            st.metric("Avg Building Size", "N/A")
    with col4:
        unique_cities = df['City'].nunique() if 'City' in df.columns else 0
        st.metric("Unique Cities", unique_cities)
    
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("Filter Controls")
    
    # Property Type Filter (only show if Property Type column exists)
    if 'Property Type' in df.columns:
        st.sidebar.subheader("1. Property Type")
        industrial_keywords = st.sidebar.multiselect(
            "Select Industrial Keywords",
            options=['industrial', 'warehouse', 'distribution', 'flex',
                     'manufacturing', 'storage', 'logistics', 'light industrial'],
            default=['industrial', 'warehouse', 'distribution', 'flex']
        )
    else:
        st.sidebar.warning("Property Type column not found in uploaded data")
        industrial_keywords = []
    
    # Building Size Filter
    st.sidebar.subheader("2. Building Size (sqft)")
    
    sqft_stats = get_column_stats(df, 'Building SqFt')
    if sqft_stats:
        # Set reasonable defaults if data is invalid
        min_val = max(int(sqft_stats['min']) if not pd.isna(sqft_stats['min']) else 0, 0)
        max_val = min(int(sqft_stats['max']) if not pd.isna(sqft_stats['max']) else 500000, 500000)
        
        if min_val >= max_val:
            min_val = 0
            max_val = 500000
        
        size_range = st.sidebar.slider(
            "Building Size Range",
            min_value=min_val,
            max_value=max_val,
            value=(min(20000, max_val), min(100000, max_val)),
            step=5000,
            format="%d sqft"
        )
    else:
        st.sidebar.warning("Building SqFt column not available")
        size_range = (0, 500000)
    
    # Lot Size Filter
    st.sidebar.subheader("3. Lot Size (acres)")
    
    if 'Lot Size Acres' in df.columns:
        lot_stats = get_column_stats(df, 'Lot Size Acres')
        if lot_stats:
            min_lot = max(float(lot_stats['min']) if not pd.isna(lot_stats['min']) else 0.0, 0.0)
            max_lot = min(float(lot_stats['max']) if not pd.isna(lot_stats['max']) else 50.0, 50.0)
            
            lot_range = st.sidebar.slider(
                "Lot Size Range",
                min_value=min_lot,
                max_value=max_lot,
                value=(min(0.5, max_lot), min(20.0, max_lot)),
                step=0.5,
                format="%.1f acres"
            )
        else:
            lot_range = (0.0, 50.0)
    else:
        st.sidebar.info("Lot Size Acres column not available")
        lot_range = (0.0, 50.0)
    
    # Price Filter
    st.sidebar.subheader("4. Sale Price")
    
    if 'Sold Price' in df.columns and not df['Sold Price'].isna().all():
        # Handle categorical data by converting to numeric first
        if df['Sold Price'].dtype.name == 'category':
            price_series = pd.to_numeric(df['Sold Price'].astype(str), errors='coerce')
        else:
            price_series = pd.to_numeric(df['Sold Price'], errors='coerce')

        # Remove any NaN values for min/max calculation
        price_series = price_series.dropna()

        if len(price_series) > 0:
            min_price = price_series.min()
            max_price = price_series.max()
        else:
            min_price, max_price = 0, 10000000
        
        # Set reasonable bounds
        max_price_val = min(int(max_price) if not pd.isna(max_price) else 10000000, 10000000)
        
        price_range = st.sidebar.slider(
            "Sale Price Range",
            min_value=0,
            max_value=max_price_val,
            value=(min(150000, max_price_val), min(2000000, max_price_val)),
            step=50000,
            format="$%d"
        )
        use_price_filter = st.sidebar.checkbox("Apply Price Filter", value=True)
    else:
        st.sidebar.info("Sale price data not available")
        use_price_filter = False
        price_range = (0, 0)
    
    # Year Built Filter
    st.sidebar.subheader("5. Year Built")
    
    if 'Year Built' in df.columns and not df['Year Built'].isna().all():
        # Handle categorical data by converting to numeric first
        if df['Year Built'].dtype.name == 'category':
            year_series = pd.to_numeric(df['Year Built'].astype(str), errors='coerce')
        else:
            year_series = pd.to_numeric(df['Year Built'], errors='coerce')

        # Remove any NaN values for min/max calculation
        year_series = year_series.dropna()

        if len(year_series) > 0:
            min_year = int(year_series.min()) if not pd.isna(year_series.min()) else 1900
            max_year = int(year_series.max()) if not pd.isna(year_series.max()) else 2025
        else:
            min_year, max_year = 1900, 2025
        
        year_range = st.sidebar.slider(
            "Year Built Range",
            min_value=min_year,
            max_value=max_year,
            value=(1980, max_year),
            step=1
        )
        use_year_filter = st.sidebar.checkbox("Apply Year Filter", value=False)
    else:
        use_year_filter = False
        year_range = (1900, 2025)
    
    # Advanced Filters
    with st.sidebar.expander("Advanced Filters"):
        # Occupancy Filter
        if 'Occupancy' in df.columns:
            occupancy_range = st.slider(
                "Occupancy Rate (%)",
                min_value=0,
                max_value=100,
                value=(0, 100),
                step=5
            )
            use_occupancy_filter = st.checkbox("Apply Occupancy Filter", value=False)
        else:
            use_occupancy_filter = False
            occupancy_range = (0, 100)
        
        # County Filter
        counties = get_unique_values(df, 'County')
        selected_counties = st.multiselect(
            "Select Counties",
            options=counties,
            default=counties
        )
        
        # State Filter
        states = get_unique_values(df, 'State')
        selected_states = st.multiselect(
            "Select States",
            options=states,
            default=states
        )
    
    # Apply Filters Button
    st.sidebar.markdown("---")
    apply_filters_btn = st.sidebar.button("Apply Filters", type="primary", use_container_width=True)
    
    # Apply filters when button is clicked or on initial load
    if apply_filters_btn or 'filtered_df' not in st.session_state:
        filter_params = {
            'industrial_keywords': industrial_keywords,
            'size_range': size_range,
            'lot_range': lot_range,
            'price_range': price_range,
            'use_price_filter': use_price_filter,
            'year_range': year_range,
            'use_year_filter': use_year_filter,
            'occupancy_range': occupancy_range,
            'use_occupancy_filter': use_occupancy_filter,
            'selected_counties': selected_counties,
            'selected_states': selected_states
        }
        
        # Validate ranges
        if validate_filter_ranges(size_range, lot_range, price_range, year_range, occupancy_range):
            with st.spinner("Applying filters..."):
                filtered_df = apply_filters(df, filter_params)
                st.session_state.filtered_df = filtered_df
                st.session_state.filter_applied = True
                st.session_state.filter_params = filter_params  # Cache filter params
            
            if apply_filters_btn:  # Only show success message when button is clicked
                st.success(f"Filters applied! Found {len(filtered_df):,} matching properties.")
    
    # Display filtered results metrics if available
    if st.session_state.filter_applied and st.session_state.filtered_df is not None:
        filtered_df = st.session_state.filtered_df
        
        # Results metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta_text = f"{len(filtered_df)/len(df)*100:.1f}% of total"
            st.metric(
                "Filtered Properties", 
                f"{len(filtered_df):,}",
                delta=delta_text
            )
        with col2:
            from utils.data_type_utils import safe_mean_calculation
            avg_filtered_sqft = safe_mean_calculation(filtered_df['Building SqFt'], 'Building SqFt')
            if not pd.isna(avg_filtered_sqft):
                st.metric("Avg Filtered Size", f"{avg_filtered_sqft:,.0f} sqft")
            else:
                st.metric("Avg Filtered Size", "N/A")
        with col3:
            if 'Sold Price' in filtered_df.columns:
                avg_price = safe_mean_calculation(filtered_df['Sold Price'], 'Sold Price')
                if not pd.isna(avg_price):
                    st.metric("Avg Sale Price", f"${avg_price:,.0f}")
                else:
                    st.metric("Avg Sale Price", "N/A")
            else:
                st.metric("Avg Sale Price", "N/A")
        with col4:
            filtered_cities = filtered_df['City'].nunique() if 'City' in filtered_df.columns else 0
            st.metric("Unique Cities", filtered_cities)
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "Analytics", "Geographic Distribution", "Export"])
        
        with tab1:
            st.subheader("Filtered Properties")
            
            # Column selection
            display_columns = st.multiselect(
                "Select columns to display",
                options=filtered_df.columns.tolist(),
                default=['Property Name', 'Property Type', 'Address', 'City', 'State',
                         'Building SqFt', 'Lot Size Acres', 'Year Built'] if all(col in filtered_df.columns for col in ['Property Name', 'Property Type', 'Address', 'City', 'State', 'Building SqFt', 'Lot Size Acres', 'Year Built']) else filtered_df.columns.tolist()[:8]
            )
            
            if display_columns:
                # Display dataframe with pagination (limit to 100 rows)
                display_df = filtered_df[display_columns].head(100)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                st.caption(f"Showing top 100 of {len(filtered_df):,} filtered properties")
            else:
                st.warning("Please select at least one column to display.")
        
        with tab2:
            st.subheader("Property Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Building size distribution
                if not filtered_df['Building SqFt'].isna().all():
                    fig = px.histogram(
                        filtered_df,
                        x='Building SqFt',
                        nbins=30,
                        title='Building Size Distribution',
                        labels={'Building SqFt': 'Building Size (sqft)', 'count': 'Number of Properties'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No building size data available for visualization.")
            
            with col2:
                # Property type distribution
                if not filtered_df['Property Type'].isna().all():
                    type_counts = filtered_df['Property Type'].value_counts().head(10)
                    if not type_counts.empty:
                        fig = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title='Top 10 Property Types'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No property type data available for visualization.")
                else:
                    st.info("No property type data available for visualization.")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Year built distribution
                if 'Year Built' in filtered_df.columns and not filtered_df['Year Built'].isna().all():
                    fig = px.histogram(
                        filtered_df,
                        x='Year Built',
                        nbins=20,
                        title='Year Built Distribution',
                        labels={'Year Built': 'Year Built', 'count': 'Number of Properties'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No year built data available for visualization.")
            
            with col4:
                # Price distribution
                if 'Sold Price' in filtered_df.columns and not filtered_df['Sold Price'].isna().all():
                    fig = px.box(
                        filtered_df,
                        y='Sold Price',
                        title='Sale Price Distribution',
                        labels={'Sold Price': 'Sale Price ($)'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sale price data available for visualization.")
        
        with tab3:
            st.subheader("Geographic Distribution")
            
            # County distribution
            if 'County' in filtered_df.columns and not filtered_df['County'].isna().all():
                county_counts = filtered_df['County'].value_counts().head(20)
                if not county_counts.empty:
                    fig = px.bar(
                        x=county_counts.values,
                        y=county_counts.index,
                        orientation='h',
                        title='Top 20 Counties by Property Count',
                        labels={'x': 'Number of Properties', 'y': 'County'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No county data available for visualization.")
            else:
                st.info("No county data available for visualization.")
            
            # City distribution
            if 'City' in filtered_df.columns and not filtered_df['City'].isna().all():
                city_counts = filtered_df['City'].value_counts().head(20)
                if not city_counts.empty:
                    fig = px.bar(
                        x=city_counts.index,
                        y=city_counts.values,
                        title='Top 20 Cities by Property Count',
                        labels={'x': 'City', 'y': 'Number of Properties'}
                    )
                    fig.update_layout(
                        height=400,
                        xaxis={'tickangle': 45}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No city data available for visualization.")
            else:
                st.info("No city data available for visualization.")
        
        with tab4:
            st.subheader("Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Export Filtered Data")
                
                # Enhanced CSV export with metadata
                csv_data = generate_enhanced_csv_export(filtered_df, df)
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name=f"filtered_properties_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Enhanced Excel export with metadata
                excel_buffer = generate_enhanced_excel_export(filtered_df, df)
                st.download_button(
                    label="Download as Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"filtered_properties_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("### Filter Summary")
                
                filter_summary = generate_filter_summary(df, filtered_df, size_range, lot_range, price_range, year_range, occupancy_range, industrial_keywords, selected_counties, selected_states, use_price_filter, use_year_filter, use_occupancy_filter)
                
                for key, value in filter_summary.items():
                    st.text(f"{key}: {value}")
                
                # Add data quality report download
                st.markdown("---")
                if st.session_state.get('processing_report'):
                    quality_report = generate_data_quality_report()
                    st.download_button(
                        label="Download Data Quality Report",
                        data=quality_report,
                        file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="Download detailed report about data processing and quality metrics"
                    )
    
    else:
        # Show initial data table when no filters applied
        st.subheader("Property Data Preview")
        preview_cols = ['Property Name', 'Property Type', 'City', 'State', 'Building SqFt'] if all(col in df.columns for col in ['Property Name', 'Property Type', 'City', 'State', 'Building SqFt']) else df.columns.tolist()[:5]
        st.dataframe(df[preview_cols].head(20), use_container_width=True)
        st.caption("Showing first 20 properties. Apply filters to see filtered results.")

if __name__ == "__main__":
    main()