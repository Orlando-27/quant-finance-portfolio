from setuptools import setup, find_packages

setup(
    name="momentum-mean-reversion",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description="Multi-asset momentum and mean reversion strategy with regime detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0", "scipy>=1.11.0", "pandas>=2.0.0",
        "matplotlib>=3.7.0", "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
    ],
    keywords=["momentum", "mean-reversion", "time-series-momentum",
              "regime-detection", "quantitative-trading"],
)
