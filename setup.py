from setuptools import setup, find_packages

setup(
    name = "darela",
    version = "1.0",
    description = "DA Release Analysis",
    author = "Shashaank N",
    packages = find_packages(),
    install_requires = ['autograd', 'alive-progress'],
)
