"""Shared two-stage charging model for transport and relay energy resources."""

from math import isfinite


def charge_time_to_full_seconds(soc_fraction, equivalent_full_charge_seconds):
    """Return time to charge from ``soc_fraction`` to 100%.

    The 0--90% stage consumes 65% of an equivalent full-charge duration and
    the 90--100% stage consumes the remaining 35%, both linear in SOC within
    their respective stages.
    """

    if not isfinite(soc_fraction) or soc_fraction < 0.0 or soc_fraction > 1.0:
        raise ValueError("soc_fraction must be finite and within [0, 1]")
    if (
        not isfinite(equivalent_full_charge_seconds)
        or equivalent_full_charge_seconds < 0.0
    ):
        raise ValueError("equivalent_full_charge_seconds must be finite and nonnegative")
    if soc_fraction < 0.9:
        return equivalent_full_charge_seconds * (
            0.65 * (0.9 - soc_fraction) / 0.9 + 0.35
        )
    return equivalent_full_charge_seconds * 0.35 * (1.0 - soc_fraction) / 0.1

