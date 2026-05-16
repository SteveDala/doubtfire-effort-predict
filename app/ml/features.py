import numpy as np


def days_between(start, end):
    return (end - start).days


def build_features(task):
    days_avail = days_between(task.start_date, task.due_date)

    return np.array([
        task.estimated_hours,
        task.target_grade,
        days_avail
    ]).reshape(1, -1)
