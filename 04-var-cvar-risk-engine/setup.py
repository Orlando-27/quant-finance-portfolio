from setuptools import setup, find_packages
setup(name="var-cvar-risk-engine", version="1.0.0",
      author="Jose Orlando Bobadilla Fuentes",
      description="Multi-method VaR/CVaR engine with backtesting and GARCH",
      packages=find_packages(where="src"), package_dir={"": "src"},
      python_requires=">=3.9")
