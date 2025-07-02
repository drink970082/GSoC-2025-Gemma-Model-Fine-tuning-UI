"""
Services module for dependency injection and service layer.

Two-Layer DI Pattern:
1. Container layer: Self-managing service registration, setup, and cleanup
2. Service layer: Business logic services that take container as dependency
"""

from .di_container import DIContainer, get_container, get_service
from .training_service import TrainingService

__all__ = ["get_container", "get_service", "DIContainer", "TrainingService"]
