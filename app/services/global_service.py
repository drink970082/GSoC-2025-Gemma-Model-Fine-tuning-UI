from app.services.training_service import TrainingService

_training_service: TrainingService | None = None


def get_training_service() -> TrainingService:
    """
    Returns a singleton instance of the TrainingService.
    """
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service
