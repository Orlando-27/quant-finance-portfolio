"""
notifier.py
-----------
Email notification service for trade events and daily summaries.

Uses Python's built-in smtplib (TLS/STARTTLS). Requires environment variables:
    SMTP_HOST, SMTP_PORT, SMTP_SENDER, SMTP_PASS, NOTIFY_TO

Configuration is read from NotificationConfig; set NOTIFY_EMAIL=true to enable.
"""

import smtplib
import os
from email.mime.text     import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image    import MIMEImage
from datetime             import datetime
from typing               import Optional, List

import pandas as pd

from src.config import NotificationConfig
from src.utils  import get_logger, format_currency

log = get_logger(__name__)


class Notifier:
    """
    Email notification dispatcher.

    Parameters
    ----------
    cfg : NotificationConfig
        SMTP settings and enable flag.
    """

    def __init__(self, cfg: NotificationConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Notification types
    # ------------------------------------------------------------------

    def trade_alert(
        self,
        symbol:     str,
        action:     str,
        qty:        int,
        price:      float,
        signal_info: dict,
    ) -> None:
        """
        Send an email alert when a trade order is submitted.

        Parameters
        ----------
        symbol      : Ticker.
        action      : "BUY" or "SELL".
        qty         : Order quantity.
        price       : Limit price.
        signal_info : Dict with indicator details (score, confidence, etc.).
        """
        subject = f"[Trading Bot] {action} {qty} {symbol} @ {format_currency(price)}"

        body = f"""
<html><body>
<h2>Trade Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
<table border="1" cellpadding="6" style="border-collapse:collapse;">
  <tr><th>Symbol</th><td><b>{symbol}</b></td></tr>
  <tr><th>Action</th><td style="color:{'green' if action=='BUY' else 'red'};"><b>{action}</b></td></tr>
  <tr><th>Quantity</th><td>{qty:,}</td></tr>
  <tr><th>Limit Price</th><td>{format_currency(price)}</td></tr>
  <tr><th>Signal Score</th><td>{signal_info.get('score', 'N/A')}</td></tr>
  <tr><th>Confidence</th><td>{signal_info.get('confidence', 0):.2%}</td></tr>
  <tr><th>RSI</th><td>{signal_info.get('rsi', 'N/A'):.1f}</td></tr>
  <tr><th>MACD Signal</th><td>{signal_info.get('macd_signal', 'N/A')}</td></tr>
  <tr><th>BB Signal</th><td>{signal_info.get('bb_signal', 'N/A')}</td></tr>
  <tr><th>EMA Signal</th><td>{signal_info.get('ema_signal', 'N/A')}</td></tr>
</table>
<p>Mode: {'PAPER TRADING' if True else 'LIVE'}</p>
</body></html>
"""
        self._send(subject, body, html=True)

    def daily_summary(
        self,
        pnl:          float,
        orders_df:    pd.DataFrame,
        positions_df: pd.DataFrame,
        chart_path:   Optional[str] = None,
    ) -> None:
        """
        Send end-of-day portfolio summary with optional chart attachment.

        Parameters
        ----------
        pnl          : Session realised + unrealised P&L.
        orders_df    : DataFrame of all orders placed during the session.
        positions_df : Current open positions.
        chart_path   : Path to PNG dashboard image to attach.
        """
        subject = (
            f"[Trading Bot] Daily Summary {datetime.now().strftime('%Y-%m-%d')} | "
            f"P&L: {format_currency(pnl)}"
        )

        orders_html = (
            orders_df.to_html(index=False, border=1)
            if not orders_df.empty
            else "<p>No orders today.</p>"
        )
        positions_html = (
            positions_df.to_html(index=False, border=1)
            if not positions_df.empty
            else "<p>No open positions.</p>"
        )

        body = f"""
<html><body>
<h2>Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}</h2>
<h3>Session P&L: <span style="color:{'green' if pnl >= 0 else 'red'};">{format_currency(pnl)}</span></h3>

<h3>Orders Placed</h3>
{orders_html}

<h3>Open Positions</h3>
{positions_html}
</body></html>
"""
        attachments = []
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                attachments.append(("dashboard.png", f.read()))

        self._send(subject, body, html=True, attachments=attachments)

    def risk_alert(self, reason: str) -> None:
        """Send immediate alert when a risk limit is breached."""
        subject = "[Trading Bot] RISK ALERT - Trading Halted"
        body    = f"<h2>Risk Limit Breached</h2><p>{reason}</p>"
        self._send(subject, body, html=True)

    # ------------------------------------------------------------------
    # SMTP dispatch
    # ------------------------------------------------------------------

    def _send(
        self,
        subject:     str,
        body:        str,
        html:        bool = False,
        attachments: List[tuple] = None,
    ) -> None:
        """
        Internal dispatcher. Silently logs if notifications are disabled.
        """
        if not self.cfg.enabled:
            log.debug("Notifications disabled. Skipping: %s", subject)
            return

        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = subject
            msg["From"]    = self.cfg.sender
            msg["To"]      = self.cfg.recipient

            mime_body = MIMEText(body, "html" if html else "plain")
            msg.attach(mime_body)

            if attachments:
                for filename, data in attachments:
                    img = MIMEImage(data, name=filename)
                    msg.attach(img)

            with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.cfg.sender, self.cfg.password)
                server.sendmail(
                    self.cfg.sender, self.cfg.recipient, msg.as_string()
                )
            log.info("Email sent: %s", subject)

        except Exception as exc:
            log.error("Failed to send email '%s': %s", subject, exc)
