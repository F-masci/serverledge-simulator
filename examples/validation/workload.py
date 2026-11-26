"""Synthetic service-time handlers for local Serverledge validation invocations."""

import time


def synthetic(params, context):
    """Sleep for the requested service time without performing real work."""
    service_s = max(0.0, float(params.get("service_s", 0.25)))
    time.sleep(service_s)
    return "completed"


def archive(params, context):
    """Backwards-compatible alias that honors service_s when provided."""
    service_s = max(0.0, float(params.get("service_s", 0.375)))
    time.sleep(service_s)
    return "completed"
