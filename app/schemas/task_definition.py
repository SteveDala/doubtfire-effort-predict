from pydantic import BaseModel
from datetime import datetime


class TaskDefinition(BaseModel):
    estimated_hours: int
    target_grade: int
    start_date: datetime
    target_date: datetime
    due_date: datetime
