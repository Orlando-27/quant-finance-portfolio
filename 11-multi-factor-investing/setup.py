from setuptools import setup, find_packages

setup(
    name="multi-factor-investing",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description=(
        "Multi-factor investing framework with Fama-French replication, "
        "Fama-MacBeth regression, Barra risk decomposition, and ML factor timing"
    ),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0", "scipy>=1.11.0", "pandas>=2.0.0",
        "scikit-learn>=1.3.0", "statsmodels>=0.14.0",
        "matplotlib>=3.7.0", "seaborn>=0.12.0", "xgboost>=2.0.0",
        "hmmlearn>=0.3.0",
    ],
    keywords=[
        "factor-investing", "fama-french", "machine-learning",
        "smart-beta", "quantitative-finance", "risk-parity",
    ],
)
