from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

class DataLoggerInterface(ABC):
    """Abstract base class defining the interface for data logging operations."""

    @abstractmethod
    def log_data(self, data: Any, timestamp: Optional[datetime] = None) -> None:
        """
        Log data with an optional timestamp.
        
        Args:
            data: The data to be logged
            timestamp: Optional timestamp for the log entry. Defaults to current time if None.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None) -> None:
        """
        Log multiple metrics with an optional timestamp.
        
        Args:
            metrics: Dictionary containing metric names and their values
            timestamp: Optional timestamp for the log entry. Defaults to current time if None.
        """
        pass

    @abstractmethod
    def start_session(self, session_id: Optional[str] = None) -> None:
        """
        Start a new logging session.
        
        Args:
            session_id: Optional identifier for the session
        """
        pass

    @abstractmethod
    def end_session(self) -> None:
        """End the current logging session."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered data to the logging destination."""
        pass