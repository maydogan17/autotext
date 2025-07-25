#!/usr/bin/env python3
"""
AutoML Text Classification - Main Entry Point

This is the main entry point for the AutoML text classification pipeline.
Simply run this script to perform end-to-end automated machine learning
on your text classification dataset.

Usage:
    python main.py [--config path/to/config.yaml] [--device cuda/cpu] [--quick-test]
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import AutoMLPipeline
from src.utils import get_device
from src.log_manager import setup_logging_from_config, print_log_stats


def main():
    """Main entry point for the AutoML pipeline."""
    
    # Setup logging first
    setup_logging_from_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AutoML Text Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python main.py
    
    # Run with custom configuration
    python main.py --config my_config.yaml
    
    """
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file (default: configs/config.yaml)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true', 
        help='Do not save models and artifacts to disk'
    )
    
    parser.add_argument(
        '--log-stats',
        action='store_true',
        help='Show log statistics and exit'
    )
    
    args = parser.parse_args()
    
    # Handle log stats
    if args.log_stats:
        logs_dir = project_root / "logs"
        print_log_stats(logs_dir)
        return
    
    # Validate configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please ensure the configuration file exists or specify a valid path with --config")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        print(f"📋 Loading configuration: {config_path}")
        device = str(get_device())
        pipeline = AutoMLPipeline(config_path, device=device)
                
        print(f"🖥️  Using device: {pipeline.device}")
        print(f"📁 Output directory: {pipeline.output_dir}")
        print(f"🎯 Dataset: {pipeline.config.data['dataset_name']}")
        print(f"🔍 HPO trials: {pipeline.config.hpo['num_trials']}")
        print(f"📊 Optimization metric: {pipeline.config.hpo['metric']}")
        
        # Run the pipeline
        print("\n🚀 Starting AutoML Pipeline...")
        results = pipeline.run(save_artifacts=not args.no_save)
        
        # Print final summary
        print("🎉" + " "*20 + "PIPELINE COMPLETED!" + " "*20 + "🎉")
        
        pipeline.print_summary()
        
        if not args.no_save:
            print(f"\n📁 Results saved to: {pipeline.output_dir}")
            print(f"   • Configuration: {pipeline.output_dir / pipeline.config.output['config_filename']}")
            print(f"   • Full results: {pipeline.output_dir / 'pipeline_results.json'}")
            print(f"   • Summary report: {pipeline.output_dir / 'pipeline_summary.txt'}")
        
        print("\n✅ AutoML pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("\nFor debugging, you can run with --quick-test to reduce execution time")
        return 1


if __name__ == "__main__":
    sys.exit(main())
