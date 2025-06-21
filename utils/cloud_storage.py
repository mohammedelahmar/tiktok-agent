import os
import mimetypes
import logging
from pathlib import Path

# Configure logger
from utils.logger import logger

class CloudStorage:
    """Class for uploading files to cloud storage services"""
    
    @staticmethod
    def upload_to_google_drive(file_path, credentials_path, token_path=None, folder_id=None):
        """
        Upload a file to Google Drive
        
        Args:
            file_path (str): Path to the file to upload
            credentials_path (str): Path to the credentials.json file
            token_path (str): Path to the token.json file (will be created if it doesn't exist)
            folder_id (str): Optional Google Drive folder ID to upload to
            
        Returns:
            dict: File metadata including id and webViewLink if successful, None otherwise
        """
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google.auth.transport.requests import Request  # Import Request object
            
            # Google Drive scopes
            SCOPES = ['https://www.googleapis.com/auth/drive.file']  # Only upload access
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            if not os.path.exists(credentials_path):
                logger.error(f"Credentials file not found: {credentials_path}")
                return None
                
            # If token_path is not provided, use the same directory as credentials_path
            if token_path is None:
                token_path = os.path.join(os.path.dirname(credentials_path), 'token.json')
            
            # Authenticate
            creds = None
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())  # Pass a Request object here
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                    
                # Save the credentials for the next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            
            # Build the Drive service
            service = build('drive', 'v3', credentials=creds)
            
            # Prepare file metadata
            file_name = os.path.basename(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            
            file_metadata = {'name': file_name}
            if folder_id:
                file_metadata['parents'] = [folder_id]
                
            # Upload the file
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
            
            logger.info(f"Uploading '{file_name}' to Google Drive...")
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()
            
            logger.info(f"Upload successful! File ID: {file.get('id')}")
            return file
            
        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {str(e)}")
            return None
    
    @staticmethod
    def upload_to_aws_s3(file_path, bucket_name, object_name=None, aws_access_key=None, aws_secret_key=None, region=None):
        """
        Upload a file to an AWS S3 bucket
        
        Args:
            file_path (str): Path to the file to upload
            bucket_name (str): Name of the S3 bucket
            object_name (str): S3 object name. If not specified, file_name is used
            aws_access_key (str): AWS access key ID. If not provided, will use env variables or aws credentials file
            aws_secret_key (str): AWS secret access key. If not provided, will use env variables or aws credentials file
            region (str): AWS region. If not provided, will use default region or env variable
            
        Returns:
            str: S3 object URL if successful, None otherwise
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # If object_name not provided, use file_name
            if object_name is None:
                object_name = os.path.basename(file_path)
                
            # Create a session using AWS credentials
            session_kwargs = {}
            if aws_access_key and aws_secret_key:
                session_kwargs['aws_access_key_id'] = aws_access_key
                session_kwargs['aws_secret_access_key'] = aws_secret_key
                
            if region:
                session_kwargs['region_name'] = region
                
            session = boto3.Session(**session_kwargs)
            s3_client = session.client('s3')
            
            logger.info(f"Uploading '{file_path}' to S3 bucket '{bucket_name}' as '{object_name}'...")
            s3_client.upload_file(file_path, bucket_name, object_name)
            
            # Generate the URL for the uploaded object
            url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
            logger.info(f"Upload successful! URL: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Error uploading to AWS S3: {str(e)}")
            return None