from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "sentilytics",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.batch_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
