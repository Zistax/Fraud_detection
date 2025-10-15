import os
import smtplib
import logging
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger("fraud_batch.alerting")
logger.addHandler(logging.NullHandler())

def send_email_gmail(sender_email: str, sender_password: str, recipient_email: str,
                     subject: str, body: str, attachment_path: str = None,
                     smtp_host: str = None, smtp_port: int = None):
    smtp_host = smtp_host or os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(smtp_port or os.environ.get("SMTP_PORT", 587))

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)

    # allegato PDF opzionale
    if attachment_path:
        attachment_file = Path(attachment_path)
        if attachment_file.exists():
            with attachment_file.open("rb") as f:
                msg.add_attachment(f.read(),
                                   maintype="application",
                                   subtype="pdf",
                                   filename=attachment_file.name)
        else:
            logger.warning("Attachment non trovato: %s. Mail inviata senza allegato.", attachment_path)

    # invio SMTP con TLS
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logger.info("Email inviata a %s", recipient_email)
    except smtplib.SMTPException as e:
        logger.exception("Errore invio email a %s: %s", recipient_email, e)
        raise
