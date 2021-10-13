import requests
import json
import os

token = os.getenv('GITHUB_ACCESS_TOKEN')

def create_pull_request(title, head, base, body):
    url = "https://api.github.com/repos/orcasound/aifororcas-livesystem/pulls"
    headers = {"Authorization": f"token {token}"}
    data = {"title": title, "head": head, "base": base, "body": body}

    res = requests.post(url=url, data=json.dumps(data), headers=headers)
