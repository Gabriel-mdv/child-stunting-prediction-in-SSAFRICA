"""
SMS Service — Africa's Talking integration
==========================================
Sends HIGH-risk stunting alerts to a supervisor phone number via SMS.

Africa's Talking covers Nigeria, Rwanda, and Ethiopia.

SETUP:
  1. pip install africastalking
  2. Create an account at https://account.africastalking.com
  3. Set environment variables (see .env.example):
       AT_USERNAME=sandbox          # use 'sandbox' for testing
       AT_API_KEY=your_api_key
       AT_SENDER_ID=CHW_TOOL        # optional alphanumeric ID

TESTING (sandbox):
  - Set AT_USERNAME=sandbox and use the sandbox API key from your AT dashboard
  - Open https://simulator.africastalking.com to see incoming messages
  - Trigger a HIGH-risk prediction — the SMS should appear in the simulator

This module is designed to fail silently — if SMS is not configured or
the send fails, the API continues working normally.
"""

import os
import logging

logger = logging.getLogger(__name__)

_sms      = None
_enabled  = False


def _init():
    global _sms, _enabled

    username = os.environ.get('AT_USERNAME', '').strip()
    api_key  = os.environ.get('AT_API_KEY',  '').strip()

    if not username or not api_key:
        logger.info("SMS disabled — AT_USERNAME or AT_API_KEY not set. "
                    "See .env.example to enable.")
        return

    try:
        import africastalking
        africastalking.initialize(username, api_key)
        _sms     = africastalking.SMS
        _enabled = True
        mode = "SANDBOX" if username.lower() == "sandbox" else "PRODUCTION"
        logger.info(f"[OK] Africa's Talking SMS enabled ({mode})")
        print(f"[OK] Africa's Talking SMS enabled ({mode})")
    except ImportError:
        logger.warning(
            "africastalking package not installed — SMS disabled. "
            "Run: pip install africastalking"
        )
    except Exception as exc:
        logger.warning(f"SMS initialisation failed: {exc}")


_init()


def is_enabled() -> bool:
    return _enabled


def send_high_risk_alert(
    phone: str,
    age_months: float,
    probability: float,
    country: str,
    top_factor: str = "",
) -> bool:
    """
    Send a HIGH-risk stunting alert via SMS.

    Parameters
    ----------
    phone       : Supervisor phone number in international format, e.g. +2348012345678
    age_months  : Child age in months
    probability : Stunting probability as a percentage (0-100)
    country     : Country name string
    top_factor  : Optional top risk factor message

    Returns True if the SMS was sent successfully, False otherwise.
    Always silent on failure — never raises.
    """
    if not _enabled:
        return False

    if not phone or not phone.strip():
        logger.debug("SMS skipped — no supervisor phone number provided")
        return False

    phone = phone.strip()
    # Ensure international format
    if phone.startswith('0'):
        logger.warning(f"Phone number {phone} looks local-format. "
                       "Use international format, e.g. +2348012345678")

    factor_line = f"Top risk factor: {top_factor}\n" if top_factor else ""

    message = (
        f"CHW HIGH RISK ALERT\n"
        f"Country: {country}\n"
        f"Child age: {int(age_months)} months\n"
        f"Stunting probability: {probability:.0f}%\n"
        f"{factor_line}"
        f"Action: Refer to nutrition clinic within 7 days."
    )

    sender_id = os.environ.get('AT_SENDER_ID', '').strip() or None

    try:
        kwargs = {'message': message, 'recipients': [phone]}
        if sender_id:
            kwargs['senderId'] = sender_id

        response = _sms.send(**kwargs)
        logger.info(f"SMS sent to {phone}: {response}")
        return True

    except Exception as exc:
        logger.warning(f"SMS send failed to {phone}: {exc}")
        return False
