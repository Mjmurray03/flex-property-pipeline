"""
Output Manager for Scalable Multi-File Pipeline
Handles Excel export, file output management, backup, and versioning
"""

import pandas as pd
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
import json 


@dataclass
class ExportResult:
    """Result of an export operation"""
    success: bool
    file_path: str
    record_count: int
    file_size_mb: float
    export_time: float
    error_message: Optional[str] = None


class OutputManager:
    """
    Manages file output operations for the scalable multi-file pipeline.
    Handles Excel export, CSV export, backup creation, and versioning.
    """
    
    def __init__(self, base_output_dir: str = "data/exports", enable_backup: bool = True):
        """
        Initialize the OutputManager.
        
        Args:
            base_output_dir: Base directory for all output files
            enable_backup: Whether to create backup copies of existing files
        """
        self.base_output_dir = Path(base_output_dir)
        self.enable_backup = enable_backup
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self._ensure_directory_exists(self.base_output_dir)
    
    def _ensure_directory_exists(self, directory: Path) -> None:
        """Create directory if it doesn't exist"""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory}: {e}")
            raise
    
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create a backup of an existing file with timestamp.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to backup file if created, None if no backup needed
        """
        if not file_path.exists():
            return None
            
        if not self.enable_backup:
            return None
            
        try:
            # Add microseconds to ensure unique timestamps for rapid operations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            backup_path = file_path.parent / backup_name
            
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in megabytes"""
        try:
            if file_path.exists():
                size_bytes = file_path.stat().st_size
                return round(size_bytes / (1024 * 1024), 3)  # More precision for small files
            return 0.0
        except Exception:
            return 0.0
    
    def export_to_excel(self, 
                       df: pd.DataFrame, 
                       output_path: str,
                       sheet_name: str = "Flex Properties") -> ExportResult:
        """
        Export DataFrame to Excel file with automatic backup and directory creation.
        
        Args:
            df: DataFrame to export
            output_path: Path for the output Excel file
            sheet_name: Name of the Excel sheet
            
        Returns:
            ExportResult with operation details
        """
        start_time = datetime.now()
        output_file = Path(output_path)
        
        try:
            # Ensure output directory exists
            self._ensure_directory_exists(output_file.parent)
            
            # Create backup if file exists
            backup_path = self._create_backup(output_file)
            if backup_path:
                self.logger.info(f"Backed up existing file to: {backup_path}")
            
            # Export to Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Calculate metrics
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = self._get_file_size_mb(output_file)
            
            self.logger.info(f"Successfully exported {len(df)} records to {output_file}")
            self.logger.info(f"File size: {file_size} MB, Export time: {export_time:.2f}s")
            
            return ExportResult(
                success=True,
                file_path=str(output_file),
                record_count=len(df),
                file_size_mb=file_size,
                export_time=export_time
            )
            
        except Exception as e:
            export_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Failed to export to Excel: {e}"
            self.logger.error(error_msg)
            
            return ExportResult(
                success=False,
                file_path=str(output_file),
                record_count=0,
                file_size_mb=0.0,
                export_time=export_time,
                error_message=error_msg
            )
    
    def export_to_csv(self, 
                     df: pd.DataFrame, 
                     output_path: str) -> ExportResult:
        """
        Export DataFrame to CSV file with automatic backup and directory creation.
        
        Args:
            df: DataFrame to export
            output_path: Path for the output CSV file
            
        Returns:
            ExportResult with operation details
        """
        start_time = datetime.now()
        output_file = Path(output_path)
        
        try:
            # Ensure output directory exists
            self._ensure_directory_exists(output_file.parent)
            
            # Create backup if file exists
            backup_path = self._create_backup(output_file)
            if backup_path:
                self.logger.info(f"Backed up existing file to: {backup_path}")
            
            # Export to CSV
            df.to_csv(output_file, index=False)
            
            # Calculate metrics
            export_time = (datetime.now() - start_time).total_seconds()
            file_size = self._get_file_size_mb(output_file)
            
            self.logger.info(f"Successfully exported {len(df)} records to {output_file}")
            self.logger.info(f"File size: {file_size} MB, Export time: {export_time:.2f}s")
            
            return ExportResult(
                success=True,
                file_path=str(output_file),
                record_count=len(df),
                file_size_mb=file_size,
                export_time=export_time
            )
            
        except Exception as e:
            export_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Failed to export to CSV: {e}"
            self.logger.error(error_msg)
            
            return ExportResult(
                success=False,
                file_path=str(output_file),
                record_count=0,
                file_size_mb=0.0,
                export_time=export_time,
                error_message=error_msg
            )
    
    def export_master_file(self, 
                          df: pd.DataFrame,
                          base_filename: str = "all_flex_properties") -> Tuple[ExportResult, Optional[ExportResult]]:
        """
        Export master file in both Excel and CSV formats.
        
        Args:
            df: DataFrame to export
            base_filename: Base filename without extension
            
        Returns:
            Tuple of (Excel ExportResult, CSV ExportResult)
        """
        excel_path = self.base_output_dir / f"{base_filename}.xlsx"
        csv_path = self.base_output_dir / f"{base_filename}.csv"
        
        # Export to Excel
        excel_result = self.export_to_excel(df, str(excel_path))
        
        # Export to CSV
        csv_result = self.export_to_csv(df, str(csv_path))
        
        return excel_result, csv_result
    
    def create_versioned_export(self, 
                               df: pd.DataFrame,
                               base_filename: str = "all_flex_properties") -> Tuple[ExportResult, Optional[ExportResult]]:
        """
        Create versioned export files with timestamp.
        
        Args:
            df: DataFrame to export
            base_filename: Base filename without extension
            
        Returns:
            Tuple of (Excel ExportResult, CSV ExportResult)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f"{base_filename}_{timestamp}"
        
        return self.export_master_file(df, versioned_filename)
    
    def save_export_metadata(self, 
                           export_results: List[ExportResult],
                           processing_stats: Dict[str, Any]) -> str:
        """
        Save export metadata and processing statistics to JSON file.
        
        Args:
            export_results: List of export results
            processing_stats: Dictionary of processing statistics
            
        Returns:
            Path to the metadata file
        """
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "export_results": [
                {
                    "success": result.success,
                    "file_path": result.file_path,
                    "record_count": result.record_count,
                    "file_size_mb": result.file_size_mb,
                    "export_time": result.export_time,
                    "error_message": result.error_message
                }
                for result in export_results
            ],
            "processing_statistics": processing_stats
        }
        
        metadata_path = self.base_output_dir / "export_metadata.json"
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved export metadata to: {metadata_path}")
            return str(metadata_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save export metadata: {e}")
            raise
    
    def cleanup_old_backups(self, max_backups: int = 5) -> int:
        """
        Clean up old backup files, keeping only the most recent ones.
        
        Args:
            max_backups: Maximum number of backup files to keep per base filename
            
        Returns:
            Number of backup files removed
        """
        removed_count = 0
        
        try:
            # Find all backup files
            backup_files = list(self.base_output_dir.glob("*_backup_*"))
            
            # Group by base filename
            backup_groups = {}
            for backup_file in backup_files:
                # Extract base filename (everything before _backup_)
                parts = backup_file.stem.split("_backup_")
                if len(parts) >= 2:
                    base_name = parts[0]
                    if base_name not in backup_groups:
                        backup_groups[base_name] = []
                    backup_groups[base_name].append(backup_file)
            
            # Remove old backups for each group
            for base_name, backups in backup_groups.items():
                if len(backups) > max_backups:
                    # Sort by modification time (newest first)
                    backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    # Remove old backups
                    for old_backup in backups[max_backups:]:
                        try:
                            old_backup.unlink()
                            removed_count += 1
                            self.logger.debug(f"Removed old backup: {old_backup}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove backup {old_backup}: {e}")
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old backup files")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
            return 0
    
    def get_output_summary(self) -> Dict[str, Any]:
        """
        Get summary of files in the output directory.
        
        Returns:
            Dictionary with output directory summary
        """
        try:
            files = list(self.base_output_dir.glob("*"))
            
            summary = {
                "output_directory": str(self.base_output_dir),
                "total_files": len(files),
                "excel_files": len(list(self.base_output_dir.glob("*.xlsx"))),
                "csv_files": len(list(self.base_output_dir.glob("*.csv"))),
                "backup_files": len(list(self.base_output_dir.glob("*_backup_*"))),
                "metadata_files": len(list(self.base_output_dir.glob("*.json"))),
                "total_size_mb": sum(self._get_file_size_mb(f) for f in files if f.is_file())
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get output summary: {e}")
            return {"error": str(e)}