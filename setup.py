from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="my_package",
    version="0.1",
    author="Cuong Dinh",
    packages=find_packages(),
    install_requires=requirements,
)