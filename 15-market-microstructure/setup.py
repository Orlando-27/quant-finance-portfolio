"""
Market Microstructure Analysis – Package Setup

Theoretical foundations:
  Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread.
      Journal of Finance, 39(4), 1127-1139.
  Glosten, L.R. & Milgrom, P.R. (1985). Bid, Ask and Transaction Prices in a
      Specialist Market with Heterogeneously Informed Traders. JFE, 14(1), 71-100.
  Kyle, A.S. (1985). Continuous Auctions and Insider Trading. Econometrica,
      53(6), 1315-1335.
  Amihud, Y. (2002). Illiquidity and Stock Returns: Cross-Section and
      Time-Series Effects. Journal of Financial Markets, 5(1), 31-56.
  Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions.
      Journal of Risk, 3(2), 5-39.
  Easley, D., Lopez de Prado, M. & O'Hara, M. (2012). Flow Toxicity and Liquidity
      in a High-Frequency World. Review of Financial Studies, 25(5), 1457-1493.
"""
from setuptools import setup, find_packages

setup(
    name="market-microstructure",
    version="1.0.0",
    author="Jose Orlando Bobadilla Fuentes",
    description="High-Frequency Data and Market Microstructure Analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
