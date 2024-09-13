#!/usr/bin/env python

import numpy as np


def arctan(y, x):
    """
    Computes the polar coordinate of a point (x, y). Ranges from [0, 2pi)
    ---
    x   : float
    y   : float
    ---
    phi : float : Polar coordinate
    """
    raw_phi = np.arctan2(y, x)
    phi = raw_phi if raw_phi >= 0 else 2*np.pi + raw_phi
    return phi