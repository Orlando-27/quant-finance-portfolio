from setuptools import setup, find_packages

setup(
    name             = "automated-trading-bot",
    version          = "1.0.0",
    description      = "Automated Trading Bot with Interactive Brokers API",
    author           = "Jose Orlando",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = [
        "ib_insync>=0.9.86",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    entry_points     = {
        "console_scripts": ["trading-bot = main:main"]
    },
    classifiers      = [
        "Programming Language :: Python :: 3",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
