#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ev import ev_cal


def test_ev_cal():
    pv, ac, ev = 1000, 2000, 1500
    sv, spi, cv, cpi = ev_cal(pv, ac, ev)
    assert sv == 500
    assert cv == -500
    assert spi == 1.5
    assert cpi == 0.75
