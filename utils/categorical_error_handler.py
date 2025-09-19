"""
Comprehensive error handling and logging for categorical data type issues.
Provides user-friendly error messages, recovery mechanisms, and detailed logging.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback
from datetime import datetime
import json
from utils.logger import setup_logging
from app.components.notifications import ErrorHandler, NotificationManager, NotificationType


class CategoricalDataError(Exception):
    """Custom exception for categorical data type issues"""
    
    def __init__(self, message: str, column_name: str = None, operation: str = None, 
                 original_error: Exception = None, suggestions: List[str] = None):
        self.column_name = column_name
        self.operation = operation
        self.original_error = original_error
        self.suggestions = suggestions or []
        super().__init__(message)


class CategoricalErrorHandler:
    """Specialized error handler for categorical data type issues"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging('categorical_error_handler')
        self.error_log = []
        self.notification_manager = NotificationManager()
    
    def handle_categorical_error(self, error: Exception, column_name: str, operation: str, 
                               context: Dict[str, Any] = None) -> str:
        """
        Generate user-friendly error messages for categorical data issues.
        
        Args:
            error: The original exception
            column_name: Name of the column causing the error
            operation: Operation that failed (e.g., 'mean calculation', 'comparison')
            context: Additional context information
            
        Returns:
            Formatted error message with guidance
        """
        error_id = f"cat_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_log)}"
        
        # Log the error details
        error_details = {
            'error_id': error_id,
            'timestamp': datetime.now().isoformat(),
            'column_name': column_name,
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.error_log.append(error_details)
        self.logger.error(f"Categorical data error in column '{column_name}' during {operation}: {str(error)}")
        
        # Generate user-friendly message
        friendly_message = self._generate_friendly_message(error, column_name, operation)
        
        # Generate suggestions
        suggestions = self._generate_error_suggestions(error, column_name, operation)
        
        # Create comprehensive error message
        error_message = f"""
**Categorical Data Error in Column '{column_name}'**

**Issue**: {friendly_message}

**Operation**: {operation}

**Error ID**: {error_id}

**Suggestions**:
{chr(10).join(f'• {suggestion}' for suggestion in suggestions)}

**Technical Details**: {str(error)}
        """.strip()
        
        # Show notification
        self.notification_manager.error(
            message=friendly_message,
            title=f"Data Type Error - {column_name}",
            duration=8000
        )
        
        return error_message
    
    def _generate_friendly_message(self, error: Exception, column_name: str, operation: str) -> str:
        """Generate user-friendly error message based on error type"""
        
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Specific categorical data error patterns
        if "categorical" in error_str and "reduction" in error_str:
            return f"Cannot perform mathematical operations on categorical data in column '{column_name}'. The data needs to be converted to numeric format first."
        
        elif "unordered categoricals" in error_str and "compare" in error_str:
            return f"Cannot compare categorical values in column '{column_name}' using numerical operators. The categorical data needs to be converted to numeric format."
        
        elif "categorical" in error_str and "mean" in error_str:
            return f"Cannot calculate average (mean) for categorical data in column '{column_name}'. The column contains text categories instead of numbers."
        
        elif error_type == "TypeError" and "categorical" in error_str:
            return f"Data type mismatch in column '{column_name}'. The column contains categorical data but numeric data is required for {operation}."
        
        elif error_type == "ValueError" and ("numeric" in error_str or "convert" in error_str):
            return f"Cannot convert categorical values in column '{column_name}' to numbers. Some values may not be numeric."
        
        elif error_type == "KeyError":
            return f"Column '{column_name}' not found in the dataset. Please check the column name."
        
        elif error_type == "AttributeError" and "categorical" in error_str:
            return f"Invalid operation attempted on categorical data in column '{column_name}'. This operation is not supported for categorical data types."
        
        else:
            return f"An error occurred while processing column '{column_name}' during {operation}: {str(error)}"
    
    def _generate_error_suggestions(self, error: Exception, column_name: str, operation: str) -> List[str]:
        """Generate specific suggestions based on error type and context"""
        
        error_type = type(error).__name__
        error_str = str(error).lower()
        suggestions = []
        
        # Categorical reduction errors (mean, sum, etc.)
        if "categorical" in error_str and "reduction" in error_str:
            suggestions.extend([
                f"Convert column '{column_name}' to numeric format using data type conversion tools",
                "Check if the categorical values represent numbers (e.g., '100', '200')",
                "Use safe calculation functions that handle categorical data automatically",
                "Consider using count() or value_counts() instead for categorical data analysis"
            ])
        
        # Categorical comparison errors
        elif "unordered categoricals" in error_str and "compare" in error_str:
            suggestions.extend([
                f"Convert column '{column_name}' to numeric format before applying filters",
                "Use equality comparisons (==, !=) instead of ordering comparisons (>, <, >=, <=)",
                "Check if the categorical values can be converted to numbers",
                "Use categorical-specific filtering methods"
            ])
        
        # General categorical data type errors
        elif "categorical" in error_str:
            suggestions.extend([
                f"Use the data type conversion utility to convert '{column_name}' to numeric",
                "Verify that the column contains numeric values stored as categories",
                "Check the data source - the column may have been imported incorrectly",
                "Use pandas.to_numeric() with errors='coerce' to handle conversion issues"
            ])
        
        # Conversion errors
        elif error_type == "ValueError" and ("numeric" in error_str or "convert" in error_str):
            suggestions.extend([
                f"Clean the data in column '{column_name}' to remove non-numeric characters",
                "Check for currency symbols, commas, or percentage signs that need to be removed",
                "Handle missing values (NaN, null, empty strings) before conversion",
                "Use safe conversion functions that handle errors gracefully"
            ])
        
        # Missing column errors
        elif error_type == "KeyError":
            suggestions.extend([
                "Check the spelling of the column name",
                "Verify that the column exists in the current dataset",
                "Check if the column was renamed during data processing",
                "Review the column mapping configuration"
            ])
        
        # General suggestions for any categorical error
        suggestions.extend([
            "Review the data quality report for more information about column types",
            "Use the dashboard's data type validation features",
            "Contact support if the issue persists"
        ])
        
        return suggestions[:6]  # Limit to 6 most relevant suggestions
    
    def log_data_type_issue(self, column_name: str, issue_type: str, details: Dict[str, Any]) -> None:
        """
        Log data type issues for debugging and monitoring.
        
        Args:
            column_name: Name of the column with issues
            issue_type: Type of issue (e.g., 'conversion_failed', 'categorical_detected')
            details: Additional details about the issue
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'column_name': column_name,
            'issue_type': issue_type,
            'details': details,
            'session_id': getattr(self, 'session_id', 'unknown')
        }
        
        self.error_log.append(log_entry)
        
        # Log with appropriate level based on issue type
        if issue_type in ['conversion_failed', 'critical_error']:
            self.logger.error(f"Data type issue in '{column_name}': {issue_type} - {details}")
        elif issue_type in ['conversion_warning', 'partial_failure']:
            self.logger.warning(f"Data type issue in '{column_name}': {issue_type} - {details}")
        else:
            self.logger.info(f"Data type issue in '{column_name}': {issue_type} - {details}")
    
    def provide_conversion_feedback(self, conversion_report: Dict[str, Any]) -> None:
        """
        Display user feedback about data type conversions performed.
        
        Args:
            conversion_report: Report from data type conversion operations
        """
        column_name = conversion_report.get('column_name', 'Unknown')
        
        if conversion_report.get('conversion_successful', False):
            values_converted = conversion_report.get('values_converted', 0)
            values_failed = conversion_report.get('values_failed', 0)
            
            if values_failed == 0:
                # Perfect conversion
                message = f"Successfully converted all values in column '{column_name}' to numeric format."
                self.notification_manager.success(
                    message=message,
                    title="Data Conversion Successful",
                    duration=5000
                )
                self.logger.info(f"Perfect conversion for column '{column_name}': {values_converted} values converted")
                
            else:
                # Partial conversion
                success_rate = (values_converted / (values_converted + values_failed)) * 100
                message = f"Converted column '{column_name}' to numeric format. {values_converted} values converted successfully, {values_failed} values became null."
                
                self.notification_manager.warning(
                    message=message,
                    title="Partial Data Conversion",
                    duration=7000
                )
                self.logger.warning(f"Partial conversion for column '{column_name}': {success_rate:.1f}% success rate")
        
        else:
            # Conversion failed
            error_msg = conversion_report.get('errors', ['Unknown error'])[0]
            message = f"Could not convert column '{column_name}' to numeric format: {error_msg}"
            
            self.notification_manager.error(
                message=message,
                title="Data Conversion Failed",
                duration=8000
            )
            self.logger.error(f"Conversion failed for column '{column_name}': {error_msg}")
    
    def create_error_recovery_plan(self, error: Exception, column_name: str, operation: str) -> Dict[str, Any]:
        """
        Create an error recovery plan with specific steps to resolve the issue.
        
        Args:
            error: The original exception
            column_name: Name of the column causing the error
            operation: Operation that failed
            
        Returns:
            Recovery plan with steps and alternatives
        """
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        recovery_plan = {
            'error_id': f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'column_name': column_name,
            'operation': operation,
            'error_type': error_type,
            'immediate_actions': [],
            'alternative_approaches': [],
            'prevention_steps': [],
            'estimated_success_rate': 0.0
        }
        
        # Categorical reduction errors
        if "categorical" in error_str and "reduction" in error_str:
            recovery_plan['immediate_actions'] = [
                "Apply automatic data type conversion to the column",
                "Use safe calculation wrapper functions",
                "Validate conversion results before proceeding"
            ]
            recovery_plan['alternative_approaches'] = [
                "Use value_counts() to analyze categorical distribution",
                "Convert to dummy variables for analysis",
                "Use categorical-specific statistical methods"
            ]
            recovery_plan['estimated_success_rate'] = 0.85
        
        # Categorical comparison errors
        elif "unordered categoricals" in error_str and "compare" in error_str:
            recovery_plan['immediate_actions'] = [
                "Convert categorical column to numeric format",
                "Use safe comparison wrapper functions",
                "Validate numeric conversion before filtering"
            ]
            recovery_plan['alternative_approaches'] = [
                "Use categorical.isin() for membership testing",
                "Apply string-based filtering if appropriate",
                "Use categorical ordering if data supports it"
            ]
            recovery_plan['estimated_success_rate'] = 0.90
        
        # General prevention steps
        recovery_plan['prevention_steps'] = [
            "Implement data type validation during data loading",
            "Use safe operation wrappers for all mathematical operations",
            "Add data type conversion to the preprocessing pipeline",
            "Monitor data quality metrics regularly"
        ]
        
        self.logger.info(f"Created recovery plan for column '{column_name}': {recovery_plan['error_id']}")
        return recovery_plan
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors encountered"""
        if not self.error_log:
            return {'total_errors': 0, 'error_types': {}, 'affected_columns': {}}
        
        error_types = {}
        affected_columns = {}
        
        for entry in self.error_log:
            # Count error types
            error_type = entry.get('error_type', entry.get('issue_type', 'unknown'))
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count affected columns
            column = entry.get('column_name', 'unknown')
            affected_columns[column] = affected_columns.get(column, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'affected_columns': affected_columns,
            'most_problematic_column': max(affected_columns.items(), key=lambda x: x[1])[0] if affected_columns else None,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    def export_error_log(self, filepath: str = None) -> str:
        """Export error log to JSON file"""
        if filepath is None:
            filepath = f"categorical_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_errors': len(self.error_log),
            'statistics': self.get_error_statistics(),
            'error_log': self.error_log
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Error log exported to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to export error log: {str(e)}")
            raise
    
    def clear_error_log(self) -> None:
        """Clear the error log"""
        cleared_count = len(self.error_log)
        self.error_log.clear()
        self.logger.info(f"Cleared {cleared_count} error log entries")


# Global error handler instance
_global_categorical_error_handler = None

def get_categorical_error_handler() -> CategoricalErrorHandler:
    """Get or create global categorical error handler instance"""
    global _global_categorical_error_handler
    if _global_categorical_error_handler is None:
        _global_categorical_error_handler = CategoricalErrorHandler()
    return _global_categorical_error_handler


# Convenience functions for common error handling scenarios
def handle_categorical_mean_error(error: Exception, column_name: str) -> str:
    """Handle errors in mean calculation with categorical data"""
    handler = get_categorical_error_handler()
    return handler.handle_categorical_error(error, column_name, "mean calculation")


def handle_categorical_comparison_error(error: Exception, column_name: str, operator: str, value: Any) -> str:
    """Handle errors in categorical data comparison"""
    handler = get_categorical_error_handler()
    context = {'operator': operator, 'comparison_value': value}
    return handler.handle_categorical_error(error, column_name, f"comparison ({operator})", context)


def handle_categorical_conversion_error(error: Exception, column_name: str) -> str:
    """Handle errors in categorical to numeric conversion"""
    handler = get_categorical_error_handler()
    return handler.handle_categorical_error(error, column_name, "data type conversion")


def log_conversion_success(column_name: str, conversion_report: Dict[str, Any]) -> None:
    """Log successful data type conversion"""
    handler = get_categorical_error_handler()
    handler.provide_conversion_feedback(conversion_report)


def log_data_quality_issue(column_name: str, issue_type: str, details: Dict[str, Any]) -> None:
    """Log data quality issues"""
    handler = get_categorical_error_handler()
    handler.log_data_type_issue(column_name, issue_type, details)