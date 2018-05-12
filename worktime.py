#!/usr/bin/env python
# -*- coding: utf-8 -*-


def fit(best, possible, bad):
    return (best + 4 * possible + bad) / 6


if __name__ == '__main__':
    print(fit(12, 10, 7))
