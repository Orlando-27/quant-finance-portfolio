from setuptools import setup, find_packages

setup(
    name="pairs-trading-cointegration",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description=(
        "Statistical arbitrage pairs trading framework with Engle-Granger, "
        "Johansen cointegration, Ornstein-Uhlenbeck calibration, and "
        "Kalman filter adaptive hedge ratios"
    ),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0", "scipy>=1.11.0", "pandas>=2.0.0",
        "scikit-learn>=1.3.0", "statsmodels>=0.14.0",
        "matplotlib>=3.7.0", "seaborn>=0.12.0",
    ],
    keywords=[
        "pairs-trading", "cointegration", "statistical-arbitrage",
        "ornstein-uhlenbeck", "kalman-filter", "mean-reversion",
    ],
)
