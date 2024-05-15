from datetime import timedelta


def normalize_time_gap(time_gap):
    max_gap = timedelta(hours=24)
    min_gap = timedelta(seconds=1)
    
    if time_gap >= max_gap:
        return 1.0
    elif time_gap <= min_gap:
        return 0.0
    else:
        return (time_gap - min_gap) / (max_gap - min_gap)
    
