#!/usr/bin/env python
# -*- coding: utf-8 -*-


def ev_cal(pv, ac, ev):
    sv = ev - pv
    spi = ev / pv
    cv = ev - ac
    cpi = ev / ac
    return sv, spi, cv, cpi


if __name__ == '__main__':
    ev_cal(800, 1000, 700)
