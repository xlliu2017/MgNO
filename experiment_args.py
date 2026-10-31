"""Argument helpers shared by the experiment entry points."""

import argparse


def smoothing_pair(value):
    """Parse a pre/post pair, accepting both ``10`` and ``1,0``."""
    parts = value.split(",") if "," in value else list(value)
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(
            "expected pre/post counts as two digits (10) or comma-separated integers (1,0)"
        )
    return [int(part) for part in parts]
