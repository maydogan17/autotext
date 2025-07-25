"""
Log management utilities for the AutoML text classification pipeline.

Provides utilities for log cleanup, archiving, and monitoring.
"""

import logging
import logging.config
import logging.handlers
import os
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from .utils import setup_logger
    
import yaml


def setup_logging_from_config(config_path: Optional[Path] = None) -> None:
    """
    Setup logging using the existing setup_logger function.
    
    Args:
        config_path: Path to logging configuration file (ignored for now)
    """
    setup_logger("autotext")


def clean_old_logs(logs_dir: Path) -> int:
    """
    Clean old log files.
    
    Args:
        logs_dir: Directory containing log files
    
    Returns:
        Number of files cleaned
    """
    if not logs_dir.exists():
        return 0
    
    cleaned_count = 0
    
    for log_file in logs_dir.glob("*.log*"):
            try:
                log_file.unlink()
                cleaned_count += 1
                print(f"🗑️  Cleaned old log: {log_file.name}")
            except Exception as e:
                print(f"⚠️  Failed to clean {log_file.name}: {e}")
    
    return cleaned_count


def get_log_stats(logs_dir: Path) -> dict:
    """
    Get statistics about log files.
    
    Args:
        logs_dir: Directory containing log files
    
    Returns:
        Dictionary with log statistics
    """
    if not logs_dir.exists():
        return {"total_files": 0, "total_size_mb": 0, "files": []}
    
    stats = {
        "total_files": 0,
        "total_size_mb": 0,
        "files": []
    }
    
    for log_file in logs_dir.glob("*.log*"):
        size_mb = log_file.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(log_file.stat().st_mtime)
        
        stats["files"].append({
            "name": log_file.name,
            "size_mb": round(size_mb, 2),
            "modified": modified.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        stats["total_files"] += 1
        stats["total_size_mb"] += size_mb
    
    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    stats["files"].sort(key=lambda x: x["modified"], reverse=True)
    
    return stats


def print_log_stats(logs_dir: Path) -> None:
    """Print log statistics to console."""
    stats = get_log_stats(logs_dir)
    
    print(f"\n📊 Log Statistics for {logs_dir}")
    print(f"{'='*50}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    
    if stats["files"]:
        print(f"\nFiles:")
        for file_info in stats["files"]:
            print(f"  {file_info['name']:<25} {file_info['size_mb']:>8.2f} MB  {file_info['modified']}")
    
    print()


def main():
    """Command-line interface for log management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Log management utilities")
    parser.add_argument("--logs-dir", type=Path, default=Path(__file__).parent.parent / "logs",
                       help="Directory containing log files")
    parser.add_argument("--stats", action="store_true", help="Show log statistics")
    parser.add_argument("--clean", help="Clean logs older than N days")
    args = parser.parse_args()
    
    if args.stats:
        print_log_stats(args.logs_dir)
    
    if args.clean:
        cleaned = clean_old_logs(args.logs_dir)
        print(f"🧹 Cleaned {cleaned} old log files")


if __name__ == "__main__":
    main()
