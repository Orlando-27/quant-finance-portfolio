"""
Portfolio Optimization: Black-Litterman and Mean-CVaR.

References:
    Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
    Black, F., & Litterman, R. (1992). Global Portfolio Optimization. FAJ.
    Rockafellar, R.T., & Uryasev, S. (2000). Optimization of CVaR. JoR.
"""
from setuptools import setup, find_packages
setup(
    name="portfolio-optimizer-black-litterman",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description="Advanced portfolio optimization: BL, Mean-CVaR, Risk Parity",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=["numpy", "scipy", "pandas", "cvxpy", "matplotlib", "plotly"],
)
