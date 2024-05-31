


def get_time_gap_hours(time_delta):
    hours = time_delta.total_seconds() // 3600
    return min(max(int(hours), 0), 24)