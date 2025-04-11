import os
import subprocess
import openai
import requests

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]
GITHUB_SHA = os.environ["GITHUB_SHA"]

openai.api_key = OPENAI_API_KEY

def get_git_diff():
    result = subprocess.run(
        ["git", "show", GITHUB_SHA, "--unified=3", "--no-color"],
        capture_output=True,
        text=True
    )
    return result.stdout


def ask_llm(diff_text):
    prompt = f"""You are a senior software engineer. Please review the following Git diff. 
    Provide a concise review about potential bugs, style issues, or performance concerns."""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional code reviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response['choices'][0]['message']['content']


def post_github_comment(review_text):
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/commits/{GITHUB_SHA}/comments"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    data = {
        "body": f"**Automated Code Review by GPT-4:**\n\n{review_text}"
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Review comment posted successfully.")
    else:
        print("Failed to post comment:", response.text)


if __name__ == "__main__":
    diff = get_git_diff()
    if not diff.strip():
        print("No diff found.")
        exit(0)

    review = ask_llm(diff)
    print("GPT Code Review:\n", review)
    post_github_comment(review)