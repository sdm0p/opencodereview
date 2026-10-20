from unittest.mock import Mock, patch

from nodes.ingestion import ingestion_node
from state import OpenCodeReviewState
import pytest
from nodes.ingestion import IngestionError


def test_parallel_file_fetch_preserves_order_and_fork_sha():
    metadata = Mock(status_code=200)
    metadata.json.return_value = {
        'head': {'sha': 'head123', 'ref': 'feature', 'repo': {'full_name': 'fork/project'}},
        'base': {'sha': 'base123', 'ref': 'main'},
    }
    diff = Mock(text='diff payload')
    session = Mock()
    session.get.side_effect = [metadata, diff]
    files = [{'filename': name, 'status': 'modified', 'patch': '+line'} for name in ['b.py', 'a.py']]
    with patch('nodes.ingestion.resolve_token', return_value='fake'), \
         patch('nodes.ingestion.GitHubSession', return_value=session), \
         patch('nodes.ingestion._fetch_all_pages', return_value=files), \
         patch('nodes.ingestion._fetch_file_content', side_effect=lambda owner, repo, sha, path, status: f'{owner}/{repo}/{sha}/{path}'):
        result = ingestion_node(OpenCodeReviewState(repo='owner/project', pr_number=1))
    assert [item.path for item in result['changed_files']] == ['b.py', 'a.py']
    assert result['changed_files'][0].content == 'fork/project/head123/b.py'
    assert result['base_sha'] == 'base123'


def test_incremental_rejects_diverged_baseline():
    metadata = Mock(status_code=200)
    metadata.json.return_value = {
        'head': {'sha': 'head123', 'ref': 'feature', 'repo': {'full_name': 'fork/project'}},
        'base': {'sha': 'base123', 'ref': 'main'},
    }
    comparison = Mock()
    comparison.json.return_value = {'status': 'diverged', 'files': []}
    session = Mock()
    session.get.side_effect = [metadata, comparison]
    with patch('nodes.ingestion.resolve_token', return_value='fake'), \
         patch('nodes.ingestion.GitHubSession', return_value=session):
        with pytest.raises(IngestionError, match='ancestor'):
            ingestion_node(OpenCodeReviewState(repo='owner/project', pr_number=1, incremental_base_sha='a' * 40))
