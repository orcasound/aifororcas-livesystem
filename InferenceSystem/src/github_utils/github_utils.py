import requests
import json
import os

token = os.getenv('GITHUB_ACCESS_TOKEN')

def create_pull_request(title, head, base, body):
    """Creates pull request within orcasound/aifororcas-livesystem.

    Args:
        `title`: The title of pull request
        `head`: The name of the branch where your changes are implemented
        `base`: The name of the branch you want the changes pulled into
        `body`: The description or contents of the pull request
    """
    url = "https://api.github.com/repos/orcasound/aifororcas-livesystem/pulls"
    headers = {"Authorization": f"token {token}"}
    data = {"title": title, "head": head, "base": base, "body": body}

    res = requests.post(url=url, data=json.dumps(data), headers=headers)
