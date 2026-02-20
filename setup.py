from setuptools import setup, find_packages

setup(
    name='xftsim',

    version='0.3.0.dev90',

    description='Forward-time genetics simulator',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    author="Richard Border",
    author_email="border.richard@gmail.com",

    url='https://github.com/rborder/xftsim',

    packages=find_packages(include=[
        'xftsim', 'xftsim*',
    ]),

    python_requires='>=3.10',

    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "numba>=0.58",
        "xarray",
        "typer>=0.9",
        "rich",
        "pyyaml",
    ],

    extras_require={
        'legacy': [
            "sgkit",
            "nptyping",
            "funcy",
            "networkx",
            "pandas_plink",
        ],
        'grg': [
            "pygrgl",
        ],
        'docs': [
            "sphinx>=7",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
            "myst-parser",
            "nbsphinx",
            "nbconvert",
            "ipython",
        ],
        'dev': [
            "pytest",
            "pytest-timeout",
            "flake8",
            "pip-tools",
        ],
        'all': [
            "xftsim[legacy,docs,dev]",
        ],
    },

    entry_points={
        'console_scripts': [
            'xftsim=xftsim.cli:app',
        ],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
    ],

    include_package_data=True,
)
