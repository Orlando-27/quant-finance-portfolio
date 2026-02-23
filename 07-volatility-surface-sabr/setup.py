from setuptools import setup, find_packages

setup(
    name="vol-surface-sabr",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    author_email="joseorlandobf@gmail.com",
    description=(
        "Volatility surface construction with SABR, SVI, Vanna-Volga, "
        "and Dupire local volatility with arbitrage-free interpolation"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/joseorlandobf/vol-surface-sabr",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
