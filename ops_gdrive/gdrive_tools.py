"""
GDrive operations for TMAI Agent.
Provides a GoogleDriveClient class to search Drive and retrieve Google Docs content using OAuth2 environment credentials.
Only Google Docs files (`application/vnd.google-apps.document`) are supported.
"""

from typing import List, Dict, Any, Optional
import os
import io

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Load environment variables from .env
load_dotenv()

class GoogleDriveClient:
    def __init__(self, scopes: Optional[List[str]] = None):
        """
        Initialize the Google Drive client using OAuth2 credentials provided via environment variables.

        Environment Variables:
            GOOGLE_CLIENT_ID: OAuth2 client ID.
            GOOGLE_CLIENT_SECRET: OAuth2 client secret.
            GOOGLE_REFRESH_TOKEN: OAuth2 refresh token.
            GOOGLE_DRIVE_SCOPES: (Optional) Comma-separated list of scopes.

        Args:
            scopes: Optional list of scopes; if not provided, defaults or ENV is used.
        """
        env_scopes = os.getenv('GOOGLE_DRIVE_SCOPES')
        if scopes is None and env_scopes:
            scopes = [s.strip() for s in env_scopes.split(',')]
        if scopes is None:
            scopes = ['https://www.googleapis.com/auth/drive.readonly']

        client_id = os.getenv('GOOGLE_CLIENT_ID')
        client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        refresh_token = os.getenv('GOOGLE_REFRESH_TOKEN')
        if not client_id or not client_secret or not refresh_token:
            raise EnvironmentError(
                'OAuth2 credentials not found. '
                'Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN in .env.'
            )

        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=client_id,
            client_secret=client_secret,
        )
        self.service = build('drive', 'v3', credentials=credentials)

    def search_drive_files(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search Google Drive for Google Docs matching a natural language query.

        Args:
            query: Free-text query to search file names or content.
            limit: Maximum number of file metadata results to return.

        Returns:
            A list of Google Docs metadata dicts containing id, name, mimeType, and webViewLink.
        """
        try:
            response = self.service.files().list(
                q=(
                    "mimeType='application/vnd.google-apps.document' "
                    f"and (name contains '{query}' or fullText contains '{query}')"
                ),
                spaces='drive',
                corpora='allDrives',
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                fields='nextPageToken, files(id, name, mimeType, webViewLink)',
                pageSize=limit
            ).execute()
            return response.get('files', [])
        except HttpError as error:
            raise RuntimeError(f"Error searching Google Docs: {error}")

    def get_drive_file_content(self, file_id: str) -> str:
        """
        Retrieve the plain-text content of a Google Docs file by its file ID.

        Args:
            file_id: The ID of the Google Docs file to export.

        Returns:
            The full plain-text content of the Google Docs file.

        Raises:
            RuntimeError if retrieval fails or if file is not a Google Doc.
        """
        try:
            meta = self.service.files().get(
                fileId=file_id,
                fields='mimeType'
            ).execute()
            if meta.get('mimeType') != 'application/vnd.google-apps.document':
                raise RuntimeError('Only Google Docs files are supported.')

            content = self.service.files().export(
                fileId=file_id,
                mimeType='text/plain'
            ).execute()
            return content.decode('utf-8') if isinstance(content, bytes) else content

        except HttpError as error:
            raise RuntimeError(f"Error retrieving Google Doc content: {error}")
