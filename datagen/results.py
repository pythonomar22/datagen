"""
Module for handling results from the data generation process
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import os
import logging

logger = logging.getLogger(__name__)


class Results:
    """Class for storing and processing generation results"""
    
    def __init__(self, data: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Results with generated data
        
        Args:
            data: List of dictionaries containing generated examples
            metadata: Optional metadata about the generation process
        """
        self.data = data
        self.metadata = metadata or {}
        self._original_len = len(data)
        
    def __len__(self) -> int:
        """Return the number of examples in the results"""
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """Get an example by index"""
        return self.data[idx]
    
    def save(self, path: str, format: Optional[str] = None) -> None:
        """
        Save results to a file
        
        Args:
            path: Path to save the file
            format: Format to save in (jsonl, json, csv, parquet). 
                   If None, inferred from path extension.
        """
        if format is None:
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.jsonl', '.json']:
                format = 'jsonl' if ext == '.jsonl' else 'json'
            elif ext == '.csv':
                format = 'csv'
            elif ext == '.parquet':
                format = 'parquet'
            else:
                format = 'jsonl'  # Default
                path = f"{path}.jsonl"
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        if format == 'jsonl':
            with open(path, 'w') as f:
                for item in self.data:
                    f.write(json.dumps(item) + '\n')
        elif format == 'json':
            with open(path, 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'data': self.data
                }, f, indent=2)
        elif format in ['csv', 'parquet']:
            df = self.to_dataframe()
            if format == 'csv':
                df.to_csv(path, index=False)
            else:
                df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(self.data)} examples to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Results":
        """
        Load results from a file
        
        Args:
            path: Path to load from
            
        Returns:
            Results object
        """
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.jsonl':
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return cls(data)
        
        elif ext == '.json':
            with open(path, 'r') as f:
                content = json.load(f)
            if isinstance(content, dict) and 'data' in content:
                return cls(content['data'], content.get('metadata'))
            else:
                return cls(content)
                
        elif ext == '.csv':
            df = pd.read_csv(path)
            return cls(df.to_dict('records'))
            
        elif ext == '.parquet':
            df = pd.read_parquet(path)
            return cls(df.to_dict('records'))
            
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame"""
        return pd.DataFrame(self.data)
    
    def filter(self, filter_fn) -> "Results":
        """
        Filter results using a filter function
        
        Args:
            filter_fn: Function that takes an example and returns True to keep it
            
        Returns:
            New Results object with filtered data
        """
        filtered_data = [item for item in self.data if filter_fn(item)]
        filtered_results = Results(filtered_data, self.metadata.copy())
        filtered_results.metadata['filtered'] = {
            'original_count': self._original_len,
            'filtered_count': len(filtered_data),
            'removed_count': self._original_len - len(filtered_data),
        }
        return filtered_results
    
    def map(self, map_fn) -> "Results":
        """
        Apply a function to each example
        
        Args:
            map_fn: Function that takes an example and returns a transformed example
            
        Returns:
            New Results object with transformed data
        """
        mapped_data = [map_fn(item) for item in self.data]
        return Results(mapped_data, self.metadata.copy())
    
    def sample(self, n: int, seed: Optional[int] = None) -> "Results":
        """
        Sample n examples randomly
        
        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility
            
        Returns:
            New Results object with sampled data
        """
        import random
        if seed is not None:
            random.seed(seed)
            
        if n >= len(self.data):
            return self
            
        sampled_data = random.sample(self.data, n)
        return Results(sampled_data, self.metadata.copy())
    
    def extend(self, other: "Results") -> "Results":
        """
        Extend current results with another Results object
        
        Args:
            other: Another Results object
            
        Returns:
            New Results object with combined data
        """
        combined_data = self.data.copy() + other.data
        
        # Merge metadata
        combined_metadata = self.metadata.copy()
        for key, value in other.metadata.items():
            if key in combined_metadata:
                if isinstance(value, dict) and isinstance(combined_metadata[key], dict):
                    combined_metadata[key].update(value)
                else:
                    combined_metadata[key] = [combined_metadata[key], value]
            else:
                combined_metadata[key] = value
        
        return Results(combined_data, combined_metadata)
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the results"""
        summary = {
            'count': len(self.data),
            'metadata': self.metadata
        }
        
        # Add more statistics if data has certain fields
        if self.data and 'instruction' in self.data[0]:
            instr_lengths = [len(item['instruction']) for item in self.data if 'instruction' in item]
            summary['instruction_length'] = {
                'min': min(instr_lengths),
                'max': max(instr_lengths),
                'mean': sum(instr_lengths) / len(instr_lengths),
            }
            
        if self.data and 'response' in self.data[0]:
            resp_lengths = [len(item['response']) for item in self.data if 'response' in item]
            summary['response_length'] = {
                'min': min(resp_lengths),
                'max': max(resp_lengths),
                'mean': sum(resp_lengths) / len(resp_lengths),
            }
            
        return summary 