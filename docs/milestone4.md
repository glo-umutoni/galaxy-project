## Software Organization

We will use the following directory structure for our package:

team19_2023
- LICENSE
- pyproject.toml
- README.md
- setup.cfg
- workflows
    - coverage.yml
    - test.yml
- docs
- app
    - __init__.py
    - data_extraction.py
    - preprocessing.py
    - wavelength_alignment.py
    - visualization.py
    - interactive_visualization.py
    - data_augmentation.py
    - classification.py
- test
    - test_data_extraction.py
    - test_preprocessing.py
    - test_wavelength_alignment.py
    - test_visualization.py
    - test_interactive_visualization.py
    - test_data_augmentation.py
    - test_classification.py


The test suite will live in its own folder called test. We will distribute the package through Test PyPi with PEP517/518, since it is the main tool to install packages.

## Licensing

We chose the MIT license for our package because it is short and simple with abundant permissions, which is attractive given that we do not want to restrict our client from abilities such as modifying and reusing the package. We are also not concerned with patents, so we do not feel that we need a more restrictive license. 
