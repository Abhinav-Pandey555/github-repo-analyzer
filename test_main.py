import pytest
from unittest.mock import Mock, patch
from github import GithubException

from main import estimate_tokens, call_with_retry, fetch_user_repos


# ── estimate_tokens ──────────────────────────────────────────────
def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0

def test_estimate_tokens_basic():
    text = "a" * 40  # 40 chars / 4 = 10
    assert estimate_tokens(text) == 10

def test_estimate_tokens_non_string_input():
    assert estimate_tokens(12345) == len("12345") // 4


# ── call_with_retry ──────────────────────────────────────────────
def test_call_with_retry_succeeds_first_try():
    func = Mock(return_value="ok")
    result = call_with_retry(func, max_retries=3, base_delay=0)
    assert result == "ok"
    assert func.call_count == 1

def test_call_with_retry_retries_on_rate_limit():
    func = Mock(side_effect=[Exception("429 quota exceeded"), "ok"])
    with patch("main.time.sleep"), patch("main.st.toast"):
        result = call_with_retry(func, max_retries=3, base_delay=0)
    assert result == "ok"
    assert func.call_count == 2

def test_call_with_retry_raises_after_max_retries():
    func = Mock(side_effect=Exception("429 rate limit"))
    with patch("main.time.sleep"), patch("main.st.toast"):
        with pytest.raises(Exception, match="rate limit"):
            call_with_retry(func, max_retries=2, base_delay=0)
    assert func.call_count == 2

def test_call_with_retry_raises_immediately_on_non_rate_limit_error():
    func = Mock(side_effect=ValueError("bad input"))
    with pytest.raises(ValueError):
        call_with_retry(func, max_retries=3, base_delay=0)
    assert func.call_count == 1  # no retry for non-rate-limit errors


# ── fetch_user_repos ─────────────────────────────────────────────
def make_fake_repo(name="repo1", fork=False, language="Python"):
    repo = Mock()
    repo.name = name
    repo.description = "test repo"
    repo.language = language
    repo.stargazers_count = 5
    repo.forks_count = 1
    repo.open_issues_count = 0
    repo.fork = fork
    repo.html_url = f"https://github.com/user/{name}"
    return repo

@patch("main.get_github_client")
def test_fetch_user_repos_success(mock_client):
    mock_user = Mock()
    mock_user.get_repos.return_value = [make_fake_repo("repo1"), make_fake_repo("repo2")]
    mock_client.return_value.get_user.return_value = mock_user

    repos, err = fetch_user_repos("someuser")

    assert err is None
    assert len(repos) == 2
    assert repos[0]["name"] == "repo1"

@patch("main.get_github_client")
def test_fetch_user_repos_filters_forks(mock_client):
    mock_user = Mock()
    mock_user.get_repos.return_value = [make_fake_repo("real"), make_fake_repo("forked", fork=True)]
    mock_client.return_value.get_user.return_value = mock_user

    repos, err = fetch_user_repos("someuser")

    assert err is None
    assert len(repos) == 1
    assert repos[0]["name"] == "real"

@patch("main.get_github_client")
def test_fetch_user_repos_invalid_username(mock_client):
    exc = GithubException(404, "Not Found", None)
    mock_client.return_value.get_user.side_effect = exc

    repos, err = fetch_user_repos("nonexistentuser")

    assert repos is None
    assert "Invalid GitHub username" in err

@patch("main.get_github_client")
def test_fetch_user_repos_rate_limited(mock_client):
    exc = GithubException(403, "Forbidden", None)
    mock_client.return_value.get_user.side_effect = exc

    repos, err = fetch_user_repos("someuser")

    assert repos is None
    assert "rate limit" in err.lower()

@patch("main.get_github_client")
def test_fetch_user_repos_respects_max_repos_limit(mock_client):
    mock_user = Mock()
    mock_user.get_repos.return_value = [make_fake_repo(f"repo{i}") for i in range(10)]
    mock_client.return_value.get_user.return_value = mock_user

    repos, err = fetch_user_repos("someuser", max_repos=3)

    assert len(repos) == 3
