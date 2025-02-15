from label_studio_processor import LabelStudioClient
from label_studio_processor.client import AuthenticationError

def main():
    # Replace with your actual API key
    API_KEY = "db746855c4ae789b7ac4c06acde8c6f482da7c05"
    
    try:
        # Initialize client
        client = LabelStudioClient(
            url="http://localhost:8080",
            api_key=API_KEY
        )
        
        # Get projects
        projects = client.get_projects()
        
        # Print projects
        print("Successfully connected to Label Studio!")
        print("\nProjects:")
        
        if not projects:
            print("No projects found in Label Studio. Please create a project first.")
            return
            
        print(f"\nFound {len(projects)} project(s):")
        for project in projects:
            # Use get() method to safely access dictionary keys
            title = project.get('title', 'No title')
            project_id = project.get('id', 'No ID')
            created = project.get('created_at', 'No date')
            print(f"- {title} (ID: {project_id}, Created: {created})")
            
    except AuthenticationError as e:
        print(f"Authentication failed: {str(e)}")
    except ConnectionError as e:
        print(f"Connection failed: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Print more details about the error
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 