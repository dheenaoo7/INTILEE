import os
from git import Repo
import shutil
from git import Git
from dotenv import load_dotenv
from pathlib import Path
from code_parser import parse_codebase
from Backend import reprocess_and_update_embeddings


class SSHConfigError(Exception):
    """Custom exception for SSH configuration errors"""
    pass

def load_ssh_config():
    """
    Load SSH configuration from .env file
    
    Returns:
    dict: SSH configuration parameters
    
    Raises:
    SSHConfigError: If required SSH configuration is missing
    """
    load_dotenv()
    
    # Get SSH key path from environment
    ssh_key_path = os.getenv('SSH_KEY_PATH')
    if not ssh_key_path:
        raise SSHConfigError("SSH_KEY_PATH must be defined in .env file")
    
    # Expand user path (~/.) if present
    ssh_key_path = os.path.expanduser(ssh_key_path)
    
    # Verify SSH key exists
    if not os.path.exists(ssh_key_path):
        raise SSHConfigError(f"SSH key not found at {ssh_key_path}")
    
    return {
        'ssh_key_path': ssh_key_path
    }

def clone_repository(repo_url, local_dir):
    """
    Clone a repository using SSH authentication configured in .env
    
    Parameters:
    repo_url (str): SSH URL of the repository (e.g., git@bitbucket.org:organization/repo.git)
    local_dir (str): Local directory path where the repository should be cloned
    
    Returns:
    str: Path to the cloned repository
    """
    try:
        # Load SSH configuration
        ssh_config = load_ssh_config()
        ssh_key_path = ssh_config['ssh_key_path']
        
        # Create the local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Extract repository name from SSH URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        clone_dir = os.path.join(local_dir, repo_name)
        
        # Remove directory if it already exists
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        
        # Set up Git environment with SSH
        env = {
            'GIT_SSH_COMMAND': f'ssh -i {ssh_key_path}'
        }
        
        print(f"Using SSH key: {ssh_key_path}")
        print(f"Cloning repository to {clone_dir}...")
        
        # Clone with SSH
        Repo.clone_from(
            repo_url, 
            clone_dir,
            env=env
        )
        
        print("Repository cloned successfully!")
        return clone_dir
    
    except SSHConfigError as e:
        print(f"SSH Configuration Error: {str(e)}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def test_ssh_connection(repo_url):
    """
    Test SSH connection to the repository using configured SSH key
    """
    try:
        # Load SSH configuration
        ssh_config = load_ssh_config()
        ssh_key_path = ssh_config['ssh_key_path']
        
        git = Git()
        # Test SSH connection with specific key
        git.execute(['ssh', '-i', ssh_key_path, '-T', repo_url.split(':')[0]])
        return True
    except Exception as e:
        print(f"SSH connection test failed: {str(e)}")
        return False
    


# Example usage
if __name__ == "__main__":
    # Example Bitbucket repository URL (SSH)
    repo_url = "git@bitbucket.org:adjetter/mobile-configuration.git"
    destination_dir = "./github_repos"
    output_dir = "./output"
    
    try:
        # Test SSH connection first
        if test_ssh_connection(repo_url):
            print("SSH connection successful!")
            
            # Clone the repository
            repo_path = clone_repository(repo_url, destination_dir)
            repo_path = "C:\\Users\\PC\\cars_prediction".encode('utf-8').decode('utf-8')
            if repo_path:
                print(f"Repository is ready at: {repo_path}")          
                output_file_path = parse_codebase(repo_path, output_dir)
                print(f"Codebase output saved to: {output_file_path}")
                if output_file_path:
                    reprocess_and_update_embeddings(output_file_path)
                
        else:
            print("Please check your SSH configuration")
            
    except SSHConfigError as e:
        print(f"SSH Configuration Error: {str(e)}")