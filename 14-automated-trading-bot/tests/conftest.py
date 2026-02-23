"""
conftest.py
-----------
Pytest configuration: mocks ib_insync at session level so the full
test suite runs without an active IB Gateway connection.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Build a minimal ib_insync stub before any src module imports it.
# This lets all 32 tests run in CI / Cloud Shell without IB Gateway.
# ---------------------------------------------------------------------------

ib_stub = MagicMock()

# Classes used in src/
ib_stub.IB            = MagicMock
ib_stub.Stock         = MagicMock
ib_stub.LimitOrder    = MagicMock
ib_stub.MarketOrder   = MagicMock
ib_stub.StopOrder     = MagicMock
ib_stub.Trade         = MagicMock
ib_stub.util          = MagicMock()

sys.modules["ib_insync"] = ib_stub
