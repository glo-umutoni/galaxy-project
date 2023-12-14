[![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team19_2023/actions/workflows/coverage.yml/badge.svg)](https://code.harvard.edu/CS107/team19_2023/actions/workflows/coverage.yml)

[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team19_2023/actions/workflows/test.yml/badge.svg)](https://code.harvard.edu/CS107/team19_2023/actions/workflows/test.yml)

# team19_2023 - 207 Project

## contributors - Team Patapoufs&co

Alexandra Ramassamy

Helen Zhao

Gloria Umutoni

Karena Yan

Katherine Hunter

## About

This library was created to facilitate the obtainment, processing, and
visualization of astronomical data from the Sloan Digital Sky Survey.  We recommend use with the 'SpecObj' table from SDSS which contains data for stars, galaxies, and QSOs. In addition to basic preprocessing,
this library supports spectral alignment of different objects, computation 
of fractional derivatives, and classification of sky objects into the three classes, "STAR", "GALAXY", or "QSO" given training data. 
Spectral visualization as well as an interactive visualization are 
supported. 

## Installation Guide

*Patapoufs in the stars* requires at least Python version 3.9.7, as well as the following packages:
- numpy
- pandas
- matplotlib
- astroquery
- astropy
- sklearn
- differint
- scipy

### Install from PyPI

`$ pip install patapoufs_in_the_stars`
