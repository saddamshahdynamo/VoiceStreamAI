from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
import json
import csv
from datetime import datetime
import os

class DataLogger(ABC):
    """Abstract base class for data logging."""
    
    @abstractmethod
    def log_data(self, data: Dict[str, Any]) -> None:
        """Log the provided data."""
        pass

class JsonLogger(DataLogger):
    """Concrete implementation of DataLogger for JSON format."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def log_data(self, data: Dict[str, Any]) -> None:
        try:
            existing_data = []
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    existing_data = json.load(f)
            
            if not isinstance(existing_data, list):
                existing_data = []
                
            data['timestamp'] = datetime.now().isoformat()
            existing_data.append(data)
            
            with open(self.filepath, 'w') as f:
                json.dump(existing_data, f, indent=4)
        except Exception as e:
            logging.error(f"Error logging to JSON: {str(e)}")

class CsvLogger(DataLogger):
    """Concrete implementation of DataLogger for CSV format."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def log_data(self, data: Dict[str, Any]) -> None:
        try:
            data['timestamp'] = datetime.now().isoformat()
            file_exists = os.path.exists(self.filepath)
            
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
        except Exception as e:
            logging.error(f"Error logging to CSV: {str(e)}")

class DataLoggerFactory:
    """Factory class for creating different types of data loggers."""
    
    @staticmethod
    def create_logger(logger_type: str, filepath: str) -> DataLogger:
        """
        Create and return a specific type of data logger.
        
        Args:
            logger_type: Type of logger ('json' or 'csv')
            filepath: Path where the log file will be stored
            
        Returns:
            DataLogger: An instance of the specified logger type
        """
        logger_type = logger_type.lower()
        if logger_type == 'json':
            return JsonLogger(filepath)
        elif logger_type == 'csv':
            return CsvLogger(filepath)
        else:
            raise ValueError(f"Unsupported logger type: {logger_type}")