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

## PyPi

You can find the package on TestPyPi [here](https://test.pypi.org/project/patapoufsinthestars/)

## Installation Guide

### Requirements:

**Requires Python>=3.9**

*patapoufsinthestars* requires at least Python version 3.9, as well as the following packages:
- [numpy](https://numpy.org/install/)
- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [astroquery](https://astroquery.readthedocs.io/en/latest/)
- [astropy](https://docs.astropy.org/en/stable/install.html)
- [sklearn](https://scikit-learn.org/stable/install.html)
- [differint](https://pypi.org/project/differint/)
- [scipy](https://scipy.org/install/)

### Install from PyPI

`$ python3 -m pip install -i https://test.pypi.org/simple/ --extra-index-url=https://pypi.org/simple patapoufsinthestars`

## Documentation

Documentation for the package can be found [here](docs/_build/html/index.html).

## Demo

A demonstration of package functionalities can be accessed through [this Jupyter notebook](docs/demo.ipynb).

## Testing

To run the test suite from the team19_2023 directory, execute:

`$ python3 -m pytest test/` 

To run the test suite with coverage report, execute:

`$ python3 -m pytest --cov=app/ test/`