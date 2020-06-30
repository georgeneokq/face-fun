import smtplib
import ssl
import email
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class Mailer:
    def __init__(self, sender_email, sender_password, smtp_server='smtp.gmail.com'):
        self.port = 465 # For SSL
        self.context = ssl.create_default_context()
        self.smtp_server = smtp_server
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = ''

    def send(self, subject, body, attachments_file_names=[], receiver_email=None):
        if receiver_email is None:
            receiver_email = self.receiver_email

        if receiver_email == '':
            return
        
        # Form the message
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = receiver_email
        message["Subject"] = subject

        message.attach(MIMEText(body, "plain"))

        if len(attachments_file_names) > 0:
            # Attachments are file names
            for file_path in attachments_file_names:
                with open(file_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())

                    # Encode file in ASCII characters to send by email
                    encoders.encode_base64(part)

                    file_path_split_parts = file_path.split('/')
                    file_name = file_path_split_parts[len(file_path_split_parts) - 1]

                    # Add header as key/value pair to attachment part
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename = {file_name}"
                    )

                    message.attach(part)

        text = message.as_string()

        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=self.context) as server:
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, receiver_email, text)


if __name__ == '__main__':
    mailer = Mailer()