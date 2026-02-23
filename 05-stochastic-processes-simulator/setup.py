from setuptools import setup, find_packages
setup(name="stochastic-processes-simulator", version="1.0.0",
      author="Jose Orlando Bobadilla Fuentes",
      description="Simulator for Wiener, GBM, OU, CIR, Heston, Merton jump-diffusion",
      packages=find_packages(where="src"), package_dir={"": "src"},
      python_requires=">=3.9")
