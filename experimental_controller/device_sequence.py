"""Ordering for the SIYI device's raw uint16 packet sequence."""
import math


def device_sequence_delta(current, previous):
    """Recognize only small forward/backward steps across 65535 -> 0.

    Raw IDs stay unchanged in logs. A reset elsewhere, duplicate, or extended
    counter going backwards retains its ordinary nonpositive delta. Source
    timestamp order and gap limits must still be checked by the caller.
    """
    delta = current - previous
    valid_ids = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v == int(v) and 0 <= v <= 65535
                    for v in (current, previous))
    if delta < 0 and valid_ids and 0 < delta + 65536 <= 16:
        return delta + 65536
    if delta > 0 and valid_ids and 0 < 65536 - delta <= 16:
        return delta - 65536  # Late packet from before the rollover.
    return delta
