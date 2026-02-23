"""
Yield Curve Modeling and Forecasting – Package Setup

Theoretical foundations:
  Nelson, C.R. & Siegel, A.F. (1987). Parsimonious Modeling of Yield Curves.
      Journal of Business, 60(4), 473-489.
  Svensson, L.E.O. (1994). Estimating and Interpreting Forward Interest Rates:
      Sweden 1992-1994. NBER Working Paper 4871.
  Diebold, F.X. & Li, C. (2006). Forecasting the Term Structure of Government
      Bond Yields. Journal of Econometrics, 130(2), 337-364.
  Litterman, R. & Scheinkman, J. (1991). Common Factors Affecting Bond Returns.
      Journal of Fixed Income, 1(1), 54-61.
  Cox, J.C., Ingersoll, J.E. & Ross, S.R. (1985). A Theory of the Term
      Structure of Interest Rates. Econometrica, 53(2), 385-407.
"""
from setuptools import setup, find_packages

setup(
    name="yield-curve-modeling",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description="Yield Curve Modeling and Forecasting with NS/NSS, PCA, VAR",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
