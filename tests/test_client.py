import pytest
import requests
from unittest.mock import Mock, patch
from label_studio_processor.client import LabelStudioClient, AuthenticationError

class TestLabelStudioClient:
    @pytest.fixture
    def mock_client(self):
        with patch('label_studio_processor.client.Client') as mock:
            yield mock

    @pytest.fixture
    def mock_requests(self):
        with patch('label_studio_processor.client.requests') as mock:
            yield mock

    def test_init(self, mock_client):
        client = LabelStudioClient(url='http://example.com', api_key='test-key')
        mock_client.assert_called_once_with(url='http://example.com', api_key='test-key')

    def test_get_project(self, mock_client):
        # Setup
        mock_project = Mock()
        mock_client.return_value.get_project.return_value = mock_project
        client = LabelStudioClient(url='http://example.com', api_key='test-key')

        # Execute
        project = client.get_project(1)

        # Assert
        assert project == mock_project
        mock_client.return_value.get_project.assert_called_once_with(1)

    def test_get_annotations(self, mock_client):
        # Setup
        mock_project = Mock()
        mock_annotations = [{'id': 1}, {'id': 2}]
        mock_project.get_annotations.return_value = mock_annotations
        mock_client.return_value.get_project.return_value = mock_project
        client = LabelStudioClient(url='http://example.com', api_key='test-key')

        # Execute
        annotations = client.get_annotations(1)

        # Assert
        assert annotations == mock_annotations
        mock_project.get_annotations.assert_called_once()

    def test_init_requires_api_key(self):
        with pytest.raises(ValueError, match="API key is required"):
            LabelStudioClient(url='http://example.com')

    def test_verify_connection_success(self, mock_requests, mock_client):
        # Setup
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response
        
        # Execute
        client = LabelStudioClient(url='http://example.com', api_key='test-key')
        
        # Assert
        mock_requests.get.assert_called_once_with(
            'http://example.com/api/projects/',
            headers={'Authorization': 'Token test-key'}
        )
    
    def test_verify_connection_auth_error(self, mock_requests, mock_client):
        # Setup
        mock_response = Mock()
        mock_response.status_code = 401
        mock_requests.get.return_value = mock_response
        mock_requests.exceptions.RequestException = requests.exceptions.RequestException
        
        # Execute & Assert
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            LabelStudioClient(url='http://example.com', api_key='invalid-key')
    
    def test_get_projects(self, mock_requests, mock_client):
        # Setup
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'id': 1, 'title': 'Project 1'},
            {'id': 2, 'title': 'Project 2'}
        ]
        mock_requests.get.return_value = mock_response
        
        # Execute
        client = LabelStudioClient(url='http://example.com', api_key='test-key')
        projects = client.get_projects()
        
        # Assert
        assert len(projects) == 2
        assert projects[0]['title'] == 'Project 1'
        assert projects[1]['title'] == 'Project 2' 