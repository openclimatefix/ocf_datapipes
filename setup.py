""" Usual setup file for package """

# read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
install_requires = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="ocf_datapipes",
    version="3.3.59",
    license="MIT",
    description="Pytorch Datapipes built for use in Open Climate Fix's forecasting work",
    author="Jacob Bieker, Jack Kelly, Peter Dudfield, James Fulton",
    author_email="info@openclimatefix.org",
    company="Open Climate Fix Ltd",
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(),
)
