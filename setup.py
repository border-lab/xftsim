import re
from pathlib import Path

from setuptools import setup, find_packages


def read_version():
    """Read __version__ from xftsim/__init__.py without importing the package."""
    init = Path(__file__).parent / "xftsim" / "__init__.py"
    match = re.search(
        r'__version__\s*=\s*["\']([^"\']+)["\']', init.read_text()
    )
    if not match:
        raise RuntimeError("Unable to find __version__ in xftsim/__init__.py")
    return match.group(1)


setup(
    name='xftsim',

    version=read_version(),

    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    author="Richard Border",
    
    author_email="border.richard@gmail.com",
    
    packages=find_packages(include=[
        'xftsim', 'xftsim*',
    ]),
    
    install_requires = [
    "funcy",
    "networkx",
    "nptyping",
    "numba==0.56.4",
    "numpy",
    "pandas",
    "pandas_plink",
    "scipy",
    "sgkit",
    "xarray",
    ],

    include_package_data=True,
)
