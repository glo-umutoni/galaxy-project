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

*Patapoufs in the stars* requires at least Python version 3.9, as well as the following packages:
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

## Contributions

Globally, every single piece of work on that package was done in pair programming/collaboration.

Alexandra Ramassamy:
hours spent : 40-50 hours
work : I worked (both on tests and implementation) on data_extraction, preprocessing, vizualization and wavelength_alignment. I also did integration and documentation works : co-work with Karena and Katherine to generate documentation using sphinx, co-work with Karena to publish the package on pypi tests; work on the demo.ipynb.

Helen Zhao:
hours spent : 40-50 hours
work : I worked on the implementation of the data_extraction module and preprocessing module. I wrote the unit and integration tests for the data_augmentation module. I also helped with writing documentation for the modules. Outside of coding, I also helped the team by reviewing pull requests. Overall, everyone worked well together. We mostly worked together at the same time for this project so I am confident to say that everyone contributed equally. 

Karena Yan:
hours spent: 40-50 hours
work: We performed many of our tasks collaboratively, with contribution from all members. In terms of individual contributions, I was one of the main people working on tests and implementation for the two visualization modules. I played a supporting role in researching and planning out the wavelength alignment module. Additionally, I did work on documentation (revising the schematic and API) and publishing the package on PyPi.

Katherine Hunter:
hours spent : 40-50 hours
work : As mentioned, we performed most of the work collaboratively.  I worked on the tests and implementation for the classification module, wrote unit tests for the preprocessing module, and reviewed pull requests.  We worked very well together as a team. 