"""
Credit Risk Modeling -- Package Setup
Author: Jose Orlando Bobadilla Fuentes, CQF
"""

from setuptools import setup, find_packages

setup(
    name="credit-risk-modeling",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description="Structural, reduced-form, and portfolio credit risk models "
                "with CDS pricing and Credit VaR Monte Carlo engine.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
)
