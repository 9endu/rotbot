from dotenv import load_dotenv
print("python-dotenv imported successfully")
import os

# Directly set values to test functionality
os.environ['GMAIL_USER'] = 'officialrotbot@gmail.com'
os.environ['GMAIL_APP_PASSWORD'] = 'jvwkvpqrhsnwjddt'
load_dotenv()

print("User:", os.getenv('GMAIL_USER'))
print("App Password:", os.getenv('GMAIL_APP_PASSWORD'))
