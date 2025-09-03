import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Hardcoded Gmail credentials
GMAIL_USER = "vinayak.uttam2023@vitstudent.ac.in"
GMAIL_PASS = "mjexaoqcqxluwvdg"   # your Google App Password

def send_gmail(subject, body, to_email=None, attachment_path=None):
    if to_email is None:
        to_email = GMAIL_USER  # send to self by default

    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_USER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, to_email, msg.as_string())
        server.quit()

        print(f"📧 Gmail notification sent: {subject}")
    except Exception as e:
        print(f"❌ Gmail notification failed: {e}")

def notify(subject, body=None, attachment=None):
    send_gmail(subject, body or "", attachment_path=attachment)
