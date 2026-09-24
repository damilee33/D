"""Q1 answerer-defined cumulative work-time metric."""

from math import isfinite


def per_trip_work_time_seconds(drone_type, box_count, round_trip_flight_seconds):
    if int(box_count) != box_count or box_count < 0:
        raise ValueError("box_count must be a nonnegative integer")
    if not isfinite(round_trip_flight_seconds) or round_trip_flight_seconds < 0:
        raise ValueError("round_trip_flight_seconds must be nonnegative")
    for field in (
        "工位固定准备时间（s）",
        "每箱装载时间（s）",
        "接收点基础交接时间（s）",
        "每箱增加交接时间（s）",
    ):
        value = drone_type[field]
        if not isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(field))
    return (
        drone_type["工位固定准备时间（s）"]
        + box_count * drone_type["每箱装载时间（s）"]
        + round_trip_flight_seconds
        + drone_type["接收点基础交接时间（s）"]
        + box_count * drone_type["每箱增加交接时间（s）"]
    )


def cumulative_work_time_seconds(trips):
    """Sum trip work times; intentionally not a parallel makespan."""

    return sum(
        per_trip_work_time_seconds(
            trip["drone_type"], trip["box_count"], trip["round_trip_flight_seconds"]
        )
        for trip in trips
    )
