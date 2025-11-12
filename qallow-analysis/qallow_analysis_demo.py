#!/usr/bin/env python3
"""
Qallow Repository Analysis Demo

This script demonstrates the methodology used to analyze the Qallow quantum-photonic 
computing platform repository. It shows how to extract key information from GitHub
APIs and perform basic repository analysis.
"""

import requests
import json
from datetime import datetime

class QallowAnalyzer:
    def __init__(self, owner="xingxerx", repo="Qallow"):
        self.owner = owner
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"
        
    def get_repository_info(self):
        """Extract basic repository information"""
        try:
            response = requests.get(self.base_url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "name": data["name"],
                    "description": data["description"],
                    "stars": data["stargazers_count"],
                    "forks": data["forks_count"],
                    "open_issues": data["open_issues_count"],
                    "language": data["language"],
                    "license": data["license"]["name"] if data["license"] else "None",
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "size_kb": data["size"]
                }
            else:
                print(f"Error fetching repository info: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching repository info: {e}")
            return None
    
    def get_issues(self, state="open", limit=5):
        """Extract recent issues/PRs"""
        try:
            url = f"{self.base_url}/issues"
            params = {"state": state, "per_page": limit}
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                issues = response.json()
                return [
                    {
                        "number": issue["number"],
                        "title": issue["title"],
                        "state": issue["state"],
                        "created_at": issue["created_at"],
                        "updated_at": issue["updated_at"],
                        "is_pr": "pull_request" in issue
                    }
                    for issue in issues
                ]
            else:
                print(f"Error fetching issues: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception fetching issues: {e}")
            return []
    
    def get_contents(self, path=""):
        """Get repository contents for a specific path"""
        try:
            url = f"{self.base_url}/contents/{path}"
            response = requests.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching contents: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching contents: {e}")
            return None
    
    def analyze_readme(self):
        """Analyze README content"""
        try:
            url = f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/main/README.md"
            response = requests.get(url)
            
            if response.status_code == 200:
                content = response.text
                
                # Extract key sections
                sections = {
                    "has_installation": "installation" in content.lower() or "setup" in content.lower(),
                    "has_examples": "example" in content.lower(),
                    "has_contributing": "contributing" in content.lower(),
                    "has_license": "license" in content.lower(),
                    "word_count": len(content.split()),
                    "line_count": len(content.split('\n'))
                }
                
                return sections
            else:
                print(f"Error fetching README: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching README: {e}")
            return None

def main():
    """Demonstrate the analysis methodology"""
    print("Qallow Repository Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = QallowAnalyzer()
    
    # Get repository information
    print("\n1. Repository Information:")
    print("-" * 30)
    repo_info = analyzer.get_repository_info()
    if repo_info:
        print(f"Name: {repo_info['name']}")
        print(f"Stars: {repo_info['stars']}")
        print(f"Forks: {repo_info['forks']}")
        print(f"Open Issues: {repo_info['open_issues']}")
        print(f"Primary Language: {repo_info['language']}")
        print(f"License: {repo_info['license']}")
        print(f"Last Updated: {repo_info['updated_at']}")
        print(f"Repository Size: {repo_info['size_kb']} KB")
    
    # Get recent issues/PRs
    print("\n2. Recent Issues and Pull Requests:")
    print("-" * 40)
    issues = analyzer.get_issues(limit=3)
    for i, issue in enumerate(issues, 1):
        issue_type = "PR" if issue["is_pr"] else "Issue"
        print(f"{i}. [{issue_type} #{issue['number']}] {issue['title']}")
        print(f"   State: {issue['state']}, Created: {issue['created_at'][:10]}")
    
    # Analyze README
    print("\n3. README Analysis:")
    print("-" * 25)
    readme_analysis = analyzer.analyze_readme()
    if readme_analysis:
        print(f"Word Count: {readme_analysis['word_count']}")
        print(f"Line Count: {readme_analysis['line_count']}")
        print(f"Has Installation Guide: {readme_analysis['has_installation']}")
        print(f"Has Examples: {readme_analysis['has_examples']}")
        print(f"Has Contributing Guide: {readme_analysis['has_contributing']}")
        print(f"Has License Info: {readme_analysis['has_license']}")
    
    # Get repository structure
    print("\n4. Repository Structure (Root Level):")
    print("-" * 40)
    contents = analyzer.get_contents()
    if contents:
        directories = [item['name'] for item in contents if item['type'] == 'dir']
        files = [item['name'] for item in contents if item['type'] == 'file']
        
        print("Directories:")
        for dir_name in directories[:5]:  # Show first 5
            print(f"  📁 {dir_name}")
        if len(directories) > 5:
            print(f"  ... and {len(directories) - 5} more")
        
        print("Files:")
        for file_name in files[:5]:  # Show first 5
            print(f"  📄 {file_name}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("This demonstrates the methodology used to analyze")
    print("the Qallow repository for the visual presentation.")

if __name__ == "__main__":
    main()