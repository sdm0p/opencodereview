import ast
import json
import sqlite3
from pathlib import Path


def test_fallback_initializes_and_remains_on_local_storage(tmp_path):
    tree = ast.parse(Path('backend/api.py').read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'_sqlite_connect', '_save_job', '_load_job'}]
    namespace = {'sqlite3': sqlite3, 'json': json, 'Any': object,
                 'DB_PATH': str(tmp_path / 'new.db'), '_postgres_connect': lambda: None}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'backend/api.py', 'exec'), namespace)
    job = {'id': 'test', 'status': 'queued'}
    namespace['_save_job'](job)
    assert job['persistence'] == 'local_fallback'
    assert namespace['_load_job']('test')['status'] == 'queued'
    namespace['_postgres_connect'] = lambda: (_ for _ in ()).throw(AssertionError('switched storage'))
    job['status'] = 'processing'
    namespace['_save_job'](job)
    with sqlite3.connect(namespace['DB_PATH']) as conn:
        assert json.loads(conn.execute('SELECT payload FROM review_jobs').fetchone()[0])['status'] == 'processing'
