import re
from typing import Tuple, Dict, Any

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(\+?\d[\d\s().-]{7,}\d)\b")
OIB_RE = re.compile(r"\b\d{11}\b")

def redact_personal_data(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Redact common personal identifiers from text.
    Returns: (redacted_text, report)
    """
    report = {"email": 0, "phone": 0, "oib": 0}

    def _sub(pattern, repl, key, s):
        matches = pattern.findall(s)
        report[key] += len(matches) if matches else 0
        return pattern.sub(repl, s)

    out = text
    out = _sub(EMAIL_RE, "[EMAIL]", "email", out)
    out = _sub(OIB_RE, "[OIB]", "oib", out)
    out = _sub(PHONE_RE, "[PHONE]", "phone", out)
    return out, report