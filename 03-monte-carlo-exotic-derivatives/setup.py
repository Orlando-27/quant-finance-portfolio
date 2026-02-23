"""
Monte Carlo Engine for Exotic Derivatives.
References:
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
    Broadie, M., & Glasserman, P. (1996). Estimating Security Price
    Derivatives Using Simulation. Management Science.
"""
from setuptools import setup, find_packages
setup(name="monte-carlo-exotic-derivatives", version="1.0.0",
      author="Jose Orlando Bobadilla Fuentes",
      description="MC pricing: Asian, Barrier, Lookback with variance reduction",
      packages=find_packages(where="src"), package_dir={"": "src"},
      python_requires=">=3.9")
