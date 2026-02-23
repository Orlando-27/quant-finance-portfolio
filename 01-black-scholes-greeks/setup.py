"""
Setup configuration for Black-Scholes Options Pricing and Greeks Engine.

References:
    Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate
    Liabilities. Journal of Political Economy, 81(3), 637-654.
    Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option Pricing:
    A Simplified Approach. Journal of Financial Economics, 7(3), 229-263.
"""
from setuptools import setup, find_packages

setup(
    name="black-scholes-greeks-engine",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description="Production-grade Black-Scholes options pricing with Greeks and volatility surface",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=["numpy>=1.24.0", "scipy>=1.10.0", "pandas>=2.0.0",
                      "matplotlib>=3.7.0", "plotly>=5.15.0"],
    extras_require={
        "dashboard": ["streamlit>=1.28.0"],
        "data": ["yfinance>=0.2.28"],
        "dev": ["pytest>=7.4.0", "black", "flake8"],
    },
)
