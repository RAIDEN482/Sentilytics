from fastapi import APIRouter
from app.api.v1.endpoints import analyze, batch, jobs, stream, auth

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
api_router.include_router(batch.router, prefix="/batch", tags=["batch"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(stream.router, prefix="/jobs", tags=["stream"]) # Stream is under /jobs/{job_id}/stream
