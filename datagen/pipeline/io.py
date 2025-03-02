"""
IO utilities for data pipeline integration
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from pathlib import Path
import csv

from datagen.results import Results

logger = logging.getLogger(__name__)


class DataLoader:
    """Loader for seed data from various formats"""
    
    @staticmethod
    def load(path: str) -> List[Dict[str, Any]]:
        """
        Load data from a file
        
        Args:
            path: Path to the file
            
        Returns:
            List of examples
        """
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.jsonl':
            return DataLoader.load_jsonl(path)
        elif ext == '.json':
            return DataLoader.load_json(path)
        elif ext == '.csv':
            return DataLoader.load_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        """Load data from a JSONL file"""
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    @staticmethod
    def load_json(path: str) -> List[Dict[str, Any]]:
        """Load data from a JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Handle different JSON formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            raise ValueError("JSON file should contain a list of examples or a dict with 'data' key")
    
    @staticmethod
    def load_csv(path: str) -> List[Dict[str, Any]]:
        """Load data from a CSV file"""
        df = pd.read_csv(path)
        return df.to_dict('records')
    
    @staticmethod
    def load_directory(directory: str, pattern: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all supported files from a directory
        
        Args:
            directory: Directory path
            pattern: Optional glob pattern to filter files
            
        Returns:
            Dictionary mapping filenames to loaded data
        """
        from glob import glob
        
        if pattern:
            file_paths = glob(os.path.join(directory, pattern))
        else:
            # Default to common data file extensions
            extensions = ['*.jsonl', '*.json', '*.csv']
            file_paths = []
            for ext in extensions:
                file_paths.extend(glob(os.path.join(directory, ext)))
                
        result = {}
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                data = DataLoader.load(file_path)
                result[filename] = data
                logger.info(f"Loaded {len(data)} examples from {filename}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        return result


class DataExporter:
    """Exporter for synthetic data to various formats"""
    
    @staticmethod
    def export(results: Results, path: str, format: Optional[str] = None) -> None:
        """
        Export results to a file
        
        This is a wrapper around Results.save() with additional options
        
        Args:
            results: Results to export
            path: Path to save the file
            format: Format to save in (defaults to inferring from extension)
        """
        results.save(path, format)
    
    @staticmethod
    def export_for_model_training(
        results: Results, 
        output_dir: str, 
        format: str = 'jsonl',
        split: bool = True,
        train_ratio: float = 0.8
    ) -> Dict[str, str]:
        """
        Export results in a format suitable for model training
        
        Args:
            results: Results to export
            output_dir: Directory to save the files
            format: Format to save in
            split: Whether to split into train/val/test
            train_ratio: Ratio of data to use for training
            
        Returns:
            Dictionary of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not split:
            # Save as a single file
            output_path = os.path.join(output_dir, f"dataset.{format}")
            DataExporter.export(results, output_path, format)
            return {"dataset": output_path}
            
        # Split into train/val sets
        # Note: For simplicity we're just doing a random split here
        # In a real application, you'd want to be more careful with the splitting
        
        # Shuffle and split data
        import random
        
        data = results.data.copy()
        random.shuffle(data)
        
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Create Results objects for each split
        train_results = Results(train_data, results.metadata.copy())
        val_results = Results(val_data, results.metadata.copy())
        
        # Save each split
        train_path = os.path.join(output_dir, f"train.{format}")
        val_path = os.path.join(output_dir, f"val.{format}")
        
        DataExporter.export(train_results, train_path, format)
        DataExporter.export(val_results, val_path, format)
        
        # Save metadata
        metadata = {
            "total_examples": len(results),
            "train_examples": len(train_results),
            "val_examples": len(val_results),
            "train_ratio": train_ratio,
            "format": format,
            **results.metadata
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "train": train_path,
            "val": val_path,
            "metadata": metadata_path
        } 