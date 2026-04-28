from email.message import EmailMessage

def mock_email(subject="Test Subject", sender="sender@example.com", to="receiver@example.com", body="This is a test email"):
    """
    Generate a fake email in bytes format.

    Args:
        subject (str): The email subject.
        sender (str): The email sender.
        to (str): The email recipient.
        body (str): The email body.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(body)

    return ("OK", [(b"1", msg.as_bytes())])

MOCK_EMAIL_DATA = {
    "email_no_json": mock_email(),
    "email_with_json": mock_email(body='{"key": "value"}'),
    "email_broken_json": mock_email(body='{"key": "value"'),
}
