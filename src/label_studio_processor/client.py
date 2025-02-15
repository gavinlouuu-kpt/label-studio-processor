from label_studio_sdk import Client
import requests
import time

class LabelStudioClient:
    def __init__(self, url="http://localhost:8080", api_key=None):
        """Initialize connection to Label Studio.
        
        Args:
            url (str): URL of the Label Studio instance
            api_key (str): API key for authentication
        """
        if not api_key:
            raise ValueError("API key is required")
            
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Token {self.api_key}'
        }
        self.client = Client(url=url, api_key=api_key)
        
        # Verify connection
        self.verify_connection()
    
    def verify_connection(self):
        """Verify connection to Label Studio by fetching projects.
        
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If API key is invalid
        """
        try:
            response = requests.get(f"{self.url}/api/projects/", headers=self.headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 404:
                raise ConnectionError(f"Could not connect to Label Studio at {self.url}")
            else:
                raise ConnectionError(f"Error connecting to Label Studio: {str(e)}")
    
    def get_projects(self):
        """Get all projects from Label Studio.
        
        Returns:
            list: List of projects
        """
        response = requests.get(f"{self.url}/api/projects/", headers=self.headers)
        response.raise_for_status()
        data = response.json()
        
        # Debug the response
        print("DEBUG - API Response:", data)
        
        return data.get('results', [])

    def get_project(self, project_id):
        """Get a specific project.
        
        Args:
            project_id (int): Project ID in Label Studio
            
        Returns:
            Project object
        """
        response = requests.get(f"{self.url}/api/projects/{project_id}/", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_annotations(self, project_id):
        """Get all annotations for a project.
        
        Args:
            project_id (int): Project ID in Label Studio
            
        Returns:
            List of annotations
        """
        # Get tasks with annotations for the project
        response = requests.get(
            f"{self.url}/api/tasks",
            params={
                'project': project_id,
                'with_annotations': True
            },
            headers=self.headers
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract annotations from tasks
        annotations = []
        for task in data:
            if task.get('annotations'):
                for annotation in task['annotations']:
                    # Include task data with each annotation
                    annotation['task'] = {
                        'id': task['id'],
                        'data': task['data']
                    }
                    annotations.append(annotation)
        
        return annotations

    def export_annotations(self, project_id, export_format='JSON'):
        """Export annotations from a project.
        
        Args:
            project_id (int): Project ID in Label Studio
            export_format (str): Format to export (JSON, CSV, COCO, etc.)
            
        Returns:
            List of annotations in the specified format
        """
        # First try the easy export API
        response = requests.get(
            f"{self.url}/api/projects/{project_id}/export",
            params={
                'exportType': export_format,
                'download_all_tasks': True  # Include tasks without annotations
            },
            headers=self.headers
        )
        
        if response.status_code == 200:
            if export_format.upper() == 'JSON':
                return response.json()
            return response.content
            
        # If easy export fails (timeout), use snapshot API
        # 1. Create snapshot
        snapshot_response = requests.post(
            f"{self.url}/api/projects/{project_id}/exports",
            headers=self.headers
        )
        snapshot_response.raise_for_status()
        snapshot = snapshot_response.json()
        
        # 2. Wait for snapshot to be ready
        export_pk = snapshot['id']
        while True:
            status_response = requests.get(
                f"{self.url}/api/projects/{project_id}/exports/{export_pk}",
                headers=self.headers
            )
            status_response.raise_for_status()
            status = status_response.json()
            
            if status['status'] == 'completed':
                break
            elif status['status'] == 'failed':
                raise Exception("Export failed")
            
            time.sleep(1)  # Wait before checking again
            
        # 3. Download the export
        download_response = requests.get(
            f"{self.url}/api/projects/{project_id}/exports/{export_pk}/download",
            headers=self.headers
        )
        download_response.raise_for_status()
        
        if export_format.upper() == 'JSON':
            return download_response.json()
        return download_response.content

class AuthenticationError(Exception):
    pass 