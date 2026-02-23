# =============================================================================
# src/alerts/manager.py | Project 18 | Jose Orlando Bobadilla Fuentes | CQF
# Alert engine: VaR breach, drawdown, single-day loss, volatility spike
# =============================================================================
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from config.settings import ALERT_THRESHOLDS

@dataclass
class Alert:
    severity: str   # INFO | WARNING | CRITICAL
    metric:   str
    message:  str
    value:    float
    threshold:float

class AlertManager:
    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or ALERT_THRESHOLDS
        self.alerts: List[Alert] = []

    def check(self, metrics: Dict) -> List[Alert]:
        self.alerts = []
        v95 = abs(metrics.get("VaR 95% (1d)", 0)) / 100
        if v95 > self.thresholds["var_95_pct"]:
            sev = "CRITICAL" if v95 > self.thresholds["var_95_pct"] * 1.5 else "WARNING"
            self.alerts.append(Alert(sev, "VaR 95%",
                f"Daily VaR {v95*100:.2f}% exceeds threshold {self.thresholds['var_95_pct']*100:.1f}%",
                v95, self.thresholds["var_95_pct"]))

        mdd = abs(metrics.get("Max Drawdown", 0)) / 100
        if mdd > self.thresholds["max_drawdown_pct"]:
            sev = "CRITICAL" if mdd > self.thresholds["max_drawdown_pct"] * 1.5 else "WARNING"
            self.alerts.append(Alert(sev, "Max Drawdown",
                f"Drawdown {mdd*100:.1f}% exceeds threshold {self.thresholds['max_drawdown_pct']*100:.0f}%",
                mdd, self.thresholds["max_drawdown_pct"]))

        vol = metrics.get("Ann. Volatility", 0) / 100
        if vol > self.thresholds["volatility_ann"]:
            self.alerts.append(Alert("WARNING", "Volatility",
                f"Annualised vol {vol*100:.1f}% exceeds threshold {self.thresholds['volatility_ann']*100:.0f}%",
                vol, self.thresholds["volatility_ann"]))

        return self.alerts

    def summary(self) -> Dict:
        return {
            "total":    len(self.alerts),
            "critical": sum(1 for a in self.alerts if a.severity == "CRITICAL"),
            "warning":  sum(1 for a in self.alerts if a.severity == "WARNING"),
            "info":     sum(1 for a in self.alerts if a.severity == "INFO"),
        }
