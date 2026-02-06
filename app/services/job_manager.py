import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from app.schemas import JobStatus

logger = logging.getLogger(__name__)

# Constants
JOBS_FILE = Path("jobs.json")

class JobManager:
    """
    Manages job state and persistence to a JSON file.
    """
    def __init__(self):
        self.jobs: Dict[str, JobStatus] = {}
        self.load_jobs()

    def load_jobs(self):
        """Load jobs from the persistence file."""
        if JOBS_FILE.exists():
            try:
                with open(JOBS_FILE, "r") as f:
                    data = json.load(f)
                    for job_id, job_data in data.items():
                        # Validate/Convert valid data back to Pydantic models
                        try:
                            self.jobs[job_id] = JobStatus(**job_data)
                        except Exception as e:
                            logger.error(f"Failed to load job {job_id}: {e}")
                logger.info(f"Loaded {len(self.jobs)} jobs from persistence.")
            except Exception as e:
                logger.error(f"Failed to load jobs file: {e}")

    def save_jobs(self):
        """Save jobs to the persistence file."""
        try:
            # Convert Pydantic models to dicts
            data = {k: v.dict() for k, v in self.jobs.items()}
            with open(JOBS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def add_job(self, job: JobStatus):
        """Add a new job and save."""
        self.jobs[job.job_id] = job
        self.save_jobs()

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, updates: Dict):
        """Update job fields and save."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            # Update fields dynamically
            updated_data = job.dict()
            updated_data.update(updates)
            
            # Re-validate with Pydantic
            self.jobs[job_id] = JobStatus(**updated_data)
            self.save_jobs()
        else:
            logger.warning(f"Attempted to update non-existent job {job_id}")

    def get_all_jobs(self) -> Dict[str, JobStatus]:
        """Get all stored jobs."""
        # Sort by creation time (newest first)? 
        # For now just return the dict, frontend can sort.
        return self.jobs

# Global instance
job_manager = JobManager()
