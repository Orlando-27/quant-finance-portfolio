from setuptools import setup, find_packages
setup(name="deep-learning-time-series", version="1.0.0",
      author="Jose Orlando Bobadilla Fuentes",
      description="LSTM, GRU, Transformer for financial time series forecasting",
      packages=find_packages(where="src"), package_dir={"": "src"},
      python_requires=">=3.9")
