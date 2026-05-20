import email
import imaplib
import json
import logging
import re
import threading
from dataclasses import dataclass
from email.header import decode_header
from email.message import Message
from typing import Optional
from sftk.common import IMAP_SERVER, IMAP_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_ARCHIVE_FOLDER_CANDIDATES
from sftk import log_config

@dataclass
class Email:
    """Dataclass for storing email data."""
    subject: str
    sender: str
    to: str
    date: str
    body: str
    attachments: list[str]

class EmailHandler(object):
    """
    Singleton class for interacting with Email.

    This is essentially a wrapper around the imaplib.IMAP4_SSL class.
    It provides additional helper functions and cleaning to make it easier to interact with the email server.
    Email.mail is an imaplib.IMAP4_SSL object, so you can use it directly if needed.
    The Python imaplib and email docs are cursed, so this class aims to make it easier to work with emails.

    For more information on IMAP, see the RFC:
    https://datatracker.ietf.org/doc/html/rfc3501.html
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "EmailHandler":
        """
        Create a new instance of the class if one does not already exist.

        Returns:
            EmailHandler: The instance of the class.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
            return cls._instance

    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor for the EmailHandler class.

        Args:
            **kwargs: Keyword arguments for the constructor.

        This function logs into the email server and selects the inbox if the connection is not already open.
        """
        # Prevent reinitialisation
        if not self._initialised:
            self.server = kwargs.get("server", IMAP_SERVER)
            self.port = kwargs.get("port", IMAP_PORT)
            self.mail = imaplib.IMAP4_SSL(self.server, self.port)

            email = kwargs.get("email", EMAIL_USER)
            password = kwargs.get("password", EMAIL_PASS)
            mailbox = kwargs.get("mailbox", "INBOX")
            self.login(email=email, password=password, mailbox=mailbox)

            self._initialized = True  # Mark as initialized
            logging.info("Created and initialized EmailHandler.")

    def login(self, email: str, password: str, mailbox: str) -> None:
        """
        Log into the email server and select the inbox.

        Args:
            email (str): The email address to log in with.
            password (str): The password for the email address.
            mailbox (str): The mailbox to select.

        Raises:
            imaplib.IMAP4.error: If the login fails.
        """
        if res != "OK":
            logging.error(f"Failed to login for email {email}: {res}")
            raise imaplib.IMAP4.error(f"Failed to login for email {email}: {res}")

        self.set_mailbox(mailbox)
        logging.info("Logged into the inbox.")

    def search(self, charset: str = None, *criterion: str) -> tuple[str, list[str]]:
        """
        Search the inbox for emails matching the given criteria.

        Args:
            charset (str): The character set to use (default None).
            *criterion (str): The search criteria, e.g. "UNSEEN", 'SUBJECT "Hello"'

        Returns:
            list[str]: List of email IDs.

        Raises:
            imaplib.IMAP4.error: If the search fails.
        """
        criterion = self.format_criteria(*criterion)
        res, emails = self.mail.search(charset, criterion)
        if res != "OK":
            logging.error(f"Search failed: {res}")
            raise imaplib.IMAP4.error(f"Search failed: {res}")

        emails = emails[0].split() if emails else []

        return emails

    def get_mailboxes(self) -> tuple[str, list[str]]:
        """
        Get a list of mailboxes on the email server.

        Returns:
            list[str]: A list of mailboxes.

        Raises:
            imaplib.IMAP4.error: If the request fails.
        """
        res, mailboxes = self.mail.list()
        if res != "OK":
            logging.error(f"Failed to get mailboxes: {res}")
            raise imaplib.IMAP4.error(f"Failed to get mailboxes: {res}")

        return [m.decode().split(' "/" ')[1].strip('"') for m in mailboxes]

    def set_mailbox(self, mailbox: str) -> None:
        """
        Select a mailbox on the email server.

        Args:
            mailbox (str): The mailbox to select.

        Returns:
            str: Mailbox name.

        Raises:
            imaplib.IMAP4.error: If the mailbox selection fails.
            ValueError: If the mailbox is not found.
        """
        # Ensure the mailbox is in available mailboxes
        mailboxes = self.get_mailboxes()

        if mailbox.lower() not in [m.lower() for m in mailboxes]:
            logging.error(f"Mailbox '{mailbox}' not found.")
            raise ValueError(f"Mailbox '{mailbox}' not found. Please choose from: {mailboxes}")

        res, mailbox = self.mail.select(mailbox)
        if res != "OK":
            logging.error(f"Failed to select mailbox: {res}")
            raise imaplib.IMAP4.error(f"Failed to select mailbox: {res}")

        return mailbox

    def get_archive_folder(self) -> str:
        """
        Get the archive folder from the list of mailboxes.

        Returns:
            str: The archive folder name.

        Raises:
            ValueError: If the archive folder is not found.
        """
        mailboxes = [m.lower() for m in self.get_mailboxes()]

        # Try to find the first matching folder from the archive candidates
        for folder in EMAIL_ARCHIVE_FOLDER_CANDIDATES:
            if folder.lower() in mailboxes:
                logging.info(f"Archive folder found: {folder}")
                return folder

        logging.error("Archive folder not found.")
        raise ValueError(f"No archive folder found. Available mailboxes: {mailboxes}")

    def get_email_content(self, mail_id: str) -> Optional[Email]:
        """
        Get the content of an email.

        This function fetches the email content using the fetch method and returns the email body.

        Args:
            mail_id (str): The email ID to fetch.

        Returns:
            str: The email body if found, otherwise None.

        Raises:
            imaplib.IMAP4.error: If the fetch fails.
            Exception: If the email body cannot be extracted.
        """
        try:
            res, data = self.mail.fetch(mail_id, "(RFC822)")
            if res != "OK":
                logging.error(f"Failed to fetch email: {res}")
                raise imaplib.IMAP4.error(f"Failed to fetch email: {res}")
            if not data or not isinstance(data[0], tuple):
                logging.error(f"No data returned for email ID: {mail_id}")
                return None

            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)
            _email = self.to_email(msg)

            return _email
        except Exception as e:
            logging.error(f"Error retrieving email body for mail ID {mail_id}: {e}")

        return None

    def set_email_flags(self, mail_id: str, flags: str | list[str]) -> None:
        """
        Helper function to set flags using plaintext commands.

        Args:
            mail_id (str): The email ID to set flags for.
            flags (str | list[str]): The flags to set.

        Raises:
            imaplib.IMAP4.error: If the command fails.

        Returns:
            None
        """
        allowed_flags = [
            "Seen",
            "Answered",
            "Flagged",
            "Deleted",
            "Draft",
            "Recent"
        ]
        # Format to capitalise the first letter and lowercase the rest
        flags = [f.capitalize() for f in flags] if isinstance(flags, list) else [flags.capitalize()]

        # Create the flag string
        flag_str = ""
        for flag in flags:
            if flag not in allowed_flags:
                logging.error(f"Invalid flag: {flag}")
                raise ValueError(f"Invalid flag: {flag}")
            flag_str += f"\\{flag} "

        flag_str = flag_str.strip()

        res, _ = self.mail.store(mail_id, "+FLAGS", f"({flag_str})")
        if res != "OK":
            logging.error(f"Failed to set flags: {res}")
            raise imaplib.IMAP4.error(f"Failed to set flags: {res}")

    def expunge(self) -> None:
        """
        Expunge the mailbox to permanently delete emails marked for deletion.

        Literally just the expunge function from the imaplib module.

        Raises:
            imaplib.IMAP4.error: If the expunge fails.
        """
        res, _ = self.mail.expunge()
        if res != "OK":
            logging.error(f"Failed to expunge mailbox: {res}")
            raise imaplib.IMAP4.error(f"Failed to expunge mailbox: {res}")

    def move_to_archive(self, mail_id: str, delete: bool = True) -> None:
        """
        Move an email to the archive folder.

        Args:
            mail_id (str): The email ID to move.
            delete (bool): Whether to delete the email after moving (default True).

        Raises:
            imaplib.IMAP4.error: If the move fails.
            ValueError: If the archive folder is not found.
        """
        try:
            archive_folder = self.get_archive_folder()
        except ValueError:
            logging.error("Failed to get archive folder.")
            raise ValueError("Failed to get archive folder.")
        res, _ = self.mail.copy(mail_id, f'"{archive_folder}"')
        if res != "OK":
            logging.error(f"Failed to copy email to archive: {res}")
            raise imaplib.IMAP4.error(f"Failed to copy email to archive: {res}")

        if delete:
            self.set_email_flags(mail_id, "Deleted")


    @staticmethod
    def format_criteria(*criterion: str) -> str:
        """
        Format the search criteria into a string to be used in the search function.

        Args:
            *criterion (str): The search criteria.

        Returns:
            str: The formatted search criteria.
        """
        criterion_union = " ".join(criterion)
        if not criterion_union.startswith("(") and not criterion_union.endswith(")"):
            criterion_union = f"({criterion_union})"
        return criterion_union

    @staticmethod
    def extract_json_from_body(body: str) -> Optional[dict]:
        """
        Extract and validate JSON from an email body.

        This method searches for a JSON object within the email body using a
        regular expression. If a valid JSON object is found, it is parsed
        and returned as a Python dictionary. If no JSON is found or the
        format is invalid, the function logs an error and returns None.

        Args:
            body (str): The email body to search for JSON content.

        Returns:
            Optional[dict]: A dictionary representation of the JSON content, or
            None if extraction fails.
        """
        if not body:
            return None

        # Extract JSON-like content
        match = re.search(r"({.*})", body, re.DOTALL)
        if not match:
            logging.error("No JSON found in email body.")
            return None

        # Validate the json content
        try:
            json_data = json.loads(match.group(1))
            return json_data
        except json.JSONDecodeError:
            logging.error("Invalid JSON format in email body.")
            return None

    @staticmethod
    def to_email(msg: Message) -> Email:
        """
        Converts an Message object into an Email object.

        Args:
            msg (Message): The email message object.

        Returns:
            Email: The Email object.
        """
        return Email(
            subject=EmailHandler.decode_mime_header(msg["Subject"]),
            sender=msg["From"],
            to=msg["To"],
            date=msg["Date"],
            body=EmailHandler.get_email_body_from_msg(msg),
            attachments=EmailHandler.extract_attachments(msg),
        )

    @staticmethod
    def decode_mime_header(header_value: Optional[str]) -> Optional[str]:
        """
        Decodes MIME-encoded email headers.

        Takes a header value and decodes it using the email.header.decode_header function.

        Args:
            header_value (str): The header value to decode.

        Returns:
            str: The decoded header
        """
        if header_value:
            decoded_parts = decode_header(header_value)
            return " ".join(
                part.decode(encoding or "utf-8") if isinstance(part, bytes) else part
                for part, encoding in decoded_parts
            )
        return None

    @staticmethod
    def get_email_body_from_msg(msg: Message) -> str:
        """Extracts plain-text body from an email.message.Message object."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")):
                    return part.get_payload(decode=True).decode(errors="ignore").strip()
        else:
            return msg.get_payload(decode=True).decode(errors="ignore").strip()
        return "(No content found)"

    @staticmethod
    def extract_attachments(msg: Message):
        """Extracts attachment filenames from an email."""
        attachments = []
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                attachments.append(part.get_filename())
        return attachments

    def close(self) -> None:
        """
        Log out and close the IMAP connection gracefully.
        """
        if self.mail is not None and self.mail.state != "LOGOUT":
            logging.info("Logging out of the email server.")
            self.mail.logout()

    def __del__(self) -> None:
        """
        Destructor for the EmailHandler class.

        This function logs out of the email server if the connection is still open.
        """
        self.close()

    def __repr__(self) -> str:
        """
        Returns a string representation of the EmailHandler class.

        Returns:
            str: The string representation of the EmailHander class.
        """
        return f"EmailHandler({self.mail.state})"
