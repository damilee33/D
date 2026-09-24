"""Versioned provisional energy formulas.

These formulas are answerer-defined because the currently supplied problem
statement does not provide explicit horizontal-cruise and ascent formulas.
"""

from math import isfinite

KWH_JOULES = 3.6e6


def _nonnegative(name, value):
    if not isfinite(value) or value < 0:
        raise ValueError("{} must be finite and nonnegative".format(name))


def _positive(name, value):
    if not isfinite(value) or value <= 0:
        raise ValueError("{} must be finite and positive".format(name))


def equivalent_range_m(payload_kg, capacity_kg, empty_range_m, full_range_m, exponent):
    _nonnegative("payload_kg", payload_kg)
    _positive("capacity_kg", capacity_kg)
    if payload_kg > capacity_kg:
        raise ValueError("payload exceeds capacity")
    _positive("empty_range_m", empty_range_m)
    _positive("full_range_m", full_range_m)
    if full_range_m > empty_range_m:
        raise ValueError("ranges must satisfy 0 < full <= empty")
    _positive("exponent", exponent)
    fraction = payload_kg / capacity_kg
    return empty_range_m - (empty_range_m - full_range_m) * fraction ** exponent


def horizontal_energy_kwh(distance_m, payload_kg, capacity_kg, empty_range_m, full_range_m, usable_energy_kwh, exponent):
    _nonnegative("distance_m", distance_m)
    _positive("usable_energy_kwh", usable_energy_kwh)
    range_m = equivalent_range_m(
        payload_kg, capacity_kg, empty_range_m, full_range_m, exponent
    )
    return distance_m / range_m * usable_energy_kwh


def climb_energy_kwh(empty_mass_kg, payload_kg, positive_height_gain_m, efficiency, gravity_m_s2):
    _nonnegative("empty_mass_kg", empty_mass_kg)
    _nonnegative("payload_kg", payload_kg)
    _nonnegative("positive_height_gain_m", positive_height_gain_m)
    if not isfinite(efficiency) or not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    _positive("gravity_m_s2", gravity_m_s2)
    return (
        (empty_mass_kg + payload_kg)
        * gravity_m_s2
        * positive_height_gain_m
        / (efficiency * KWH_JOULES)
    )


def transport_segment_energy_kwh(distance_m, positive_height_gain_m, payload_kg, drone_type, energy_config):
    exponent = energy_config["equivalent_range_exponent"]
    gravity_m_s2 = energy_config["gravity_m_s2"]
    horizontal = horizontal_energy_kwh(
        distance_m=distance_m,
        payload_kg=payload_kg,
        capacity_kg=drone_type["最大载货质量（kg）"],
        empty_range_m=drone_type["空载标准航程（m）"],
        full_range_m=drone_type["满载标准航程（m）"],
        usable_energy_kwh=drone_type["电池可用能量（kWh）"],
        exponent=exponent,
    )
    climb = climb_energy_kwh(
        empty_mass_kg=drone_type["含电池空载总质量（kg）"],
        payload_kg=payload_kg,
        positive_height_gain_m=positive_height_gain_m,
        efficiency=drone_type["爬升能耗效率"],
        gravity_m_s2=gravity_m_s2,
    )
    return {
        "energy_model_version": energy_config["version"],
        "horizontal_kwh": horizontal,
        "climb_kwh": climb,
        "total_kwh": horizontal + climb,
    }


def relay_power_energy_kwh(cruise_power_kw, cruise_seconds, hover_power_kw, communication_power_kw, service_seconds):
    for name, value in (
        ("cruise_power_kw", cruise_power_kw),
        ("cruise_seconds", cruise_seconds),
        ("hover_power_kw", hover_power_kw),
        ("communication_power_kw", communication_power_kw),
        ("service_seconds", service_seconds),
    ):
        _nonnegative(name, value)
    return (
        cruise_power_kw * cruise_seconds
        + (hover_power_kw + communication_power_kw) * service_seconds
    ) / 3600.0
