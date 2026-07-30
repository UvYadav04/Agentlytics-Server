import asyncio
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from shared.config import get_settings

logger = logging.getLogger("shared.email")

ACCENT = "#CC785C"
TEXT = "#262624"
MUTED = "#83807A"
BORDER = "#E8E4D9"
BG = "#F5F4EE"
_DEFAULT_FROM_EMAIL = "dineshnirban01@gmail.com"


def _smtp_settings() -> dict:
    settings = get_settings()
    host = settings.get("SMTP_SERVER") or settings.get("SMTP_HOST")
    username = settings.get("SMTP_LOGIN") or settings.get("SMTP_USERNAME")
    password = settings.get("SMTP_KEY") or settings.get("SMTP_PASSWORD")
    return {
        "host": host,
        "port": int(settings.get("SMTP_PORT", "587") or "587"),
        "username": username,
        "password": password,
        "use_tls": (settings.get("SMTP_USE_TLS", "true") or "true").lower() != "false",
        "from_email": settings.get("SMTP_FROM_EMAIL") or _DEFAULT_FROM_EMAIL,
        "from_name": settings.get("SMTP_FROM_NAME") or "Agentlytics",
    }


def _frontend_url(path: str) -> str:
    origin = (get_settings().get("FRONTEND_ORIGIN") or "http://localhost:3000").rstrip("/")
    return f"{origin}{path}"


def _send_sync(to_email: str, subject: str, html_body: str, text_body: str) -> None:
    cfg = _smtp_settings()
    if not cfg["host"] or not cfg["from_email"]:
        logger.warning(
            "SMTP not configured (SMTP_HOST/SMTP_FROM_EMAIL missing) - skipping email to %s: %r",
            to_email, subject,
        )
        return

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f'{cfg["from_name"]} <{cfg["from_email"]}>'
    message["To"] = to_email
    message.attach(MIMEText(text_body, "plain"))
    message.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as server:
        if cfg["use_tls"]:
            server.starttls()
        if cfg["username"] and cfg["password"]:
            server.login(cfg["username"], cfg["password"])
        server.sendmail(cfg["from_email"], [to_email], message.as_string())


async def send_email(to_email: str, subject: str, html_body: str, text_body: str) -> None:
    """smtplib is blocking - the actual send runs in a worker thread (same asyncio.to_thread
    pattern used elsewhere for blocking I/O, e.g. worker_service's sandbox_manager calls)
    rather than stalling the event loop for however long the SMTP round trip takes. Never
    raises - see module docstring."""
    try:
        await asyncio.to_thread(_send_sync, to_email, subject, html_body, text_body)
    except Exception:
        logger.exception("failed to send email to %s (subject=%r)", to_email, subject)


def _wrap_html(preheader: str, heading: str, body_html: str, cta_label: str = None, cta_url: str = None) -> str:
    cta_html = ""
    if cta_label and cta_url:
        cta_html = f"""
        <table role="presentation" cellpadding="0" cellspacing="0" style="margin:28px 0;">
          <tr><td style="border-radius:999px;background:{ACCENT};">
            <a href="{cta_url}" style="display:inline-block;padding:12px 28px;font-size:14px;
               font-weight:600;color:#ffffff;text-decoration:none;border-radius:999px;">
              {cta_label}
            </a>
          </td></tr>
        </table>
        <p style="margin:0 0 4px;font-size:12px;color:{MUTED};">
          Or copy this link: <span style="word-break:break-all;">{cta_url}</span>
        </p>
        """
    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:32px 16px;background:{BG};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <span style="display:none;max-height:0;overflow:hidden;">{preheader}</span>
  <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
    <tr><td align="center">
      <table role="presentation" cellpadding="0" cellspacing="0" width="480"
             style="max-width:480px;background:#ffffff;border:1px solid {BORDER};
                    border-radius:16px;padding:32px;">
        <tr><td>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">
            <span style="display:inline-block;width:20px;height:20px;border-radius:6px;background:{ACCENT};"></span>
            <span style="font-size:15px;font-weight:700;color:{TEXT};">Agentlytics</span>
          </div>
          <h1 style="margin:0 0 12px;font-size:20px;font-weight:700;color:{TEXT};">{heading}</h1>
          <div style="font-size:14px;line-height:1.6;color:{TEXT};">{body_html}</div>
          {cta_html}
          <p style="margin:24px 0 0;font-size:12px;color:{MUTED};">
            Agentlytics - answers you can trace back to a real file, every time.
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


async def send_verification_email(to_email: str, name: str, token: str) -> None:
    expire_hours = get_settings().get("EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS", "24") or "24"
    link = _frontend_url(f"/verify-email?token={token}")
    subject = "Verify your email for Agentlytics"
    html = _wrap_html(
        preheader="Confirm your email to finish setting up your account.",
        heading="Confirm your email",
        body_html=(
            f"<p>Hi {name},</p>"
            f"<p>Click below to verify <strong>{to_email}</strong> and finish setting up your "
            f"Agentlytics account. This link expires in {expire_hours} hours.</p>"
        ),
        cta_label="Verify email",
        cta_url=link,
    )
    text = (
        f"Hi {name},\n\nVerify your email to finish setting up your Agentlytics account:\n{link}\n\n"
        f"This link expires in {expire_hours} hours."
    )
    await send_email(to_email, subject, html, text)


async def send_temporary_password_email(to_email: str, name: str, temporary_password: str) -> None:
    profile_link = _frontend_url("/profile")
    subject = "Your new Agentlytics password"
    html = _wrap_html(
        preheader="Here's a temporary password to get back into your account.",
        heading="Your new temporary password",
        body_html=(
            f"<p>Hi {name},</p>"
            f"<p>You (or someone with access to this inbox) requested a password reset. "
            f"Your new temporary password is:</p>"
            f"<p style=\"margin:16px 0;padding:12px 16px;background:{BG};border:1px solid {BORDER};"
            f"border-radius:8px;font-size:16px;font-weight:600;letter-spacing:0.02em;"
            f"font-family:monospace;\">{temporary_password}</p>"
            f"<p>You can change this any time from your profile.</p>"
        ),
        cta_label="Go to profile",
        cta_url=profile_link,
    )
    text = (
        f"Hi {name},\n\nYou requested a password reset. Your new temporary password is:\n\n"
        f"{temporary_password}\n\nYou can change this any time from your profile: {profile_link}"
    )
    await send_email(to_email, subject, html, text)


async def send_password_changed_email(to_email: str, name: str) -> None:
    subject = "Your Agentlytics password was changed"
    html = _wrap_html(
        preheader="Confirming your password was just changed.",
        heading="Password changed",
        body_html=(
            f"<p>Hi {name},</p>"
            f"<p>This is a confirmation that your Agentlytics account password was just changed. "
            f"If this wasn't you, reset your password immediately from the login page and "
            f"contact support.</p>"
        ),
    )
    text = (
        f"Hi {name},\n\nThis is a confirmation that your Agentlytics account password was just "
        f"changed. If this wasn't you, reset your password immediately from the login page."
    )
    await send_email(to_email, subject, html, text)
