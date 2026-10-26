"""Source activation rejects drift and foreign editable packages."""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
from tools._upstream import activate


@pytest.fixture
def tool(tmp_path):
    source = tmp_path / 'upstream'
    source.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    git('init', '-q')
    (source / 'model.py').write_text('VALUE = 1\n')
    git('add', 'model.py')
    git('-c', 'user.name=Migration test', '-c', 'user.email=migration@example.invalid', '-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'Official fixture')
    (tmp_path / 'upstream.json').write_text(json.dumps({'commit': git('rev-parse', 'HEAD')}))
    return tmp_path


def test_activates_only_verified_source(tool, monkeypatch):
    monkeypatch.setattr(sys, 'path', list(sys.path))
    assert activate(tool, ('migration_fixture',)) == tool / 'upstream'
    assert sys.path[0] == str(tool / 'upstream')


def test_rejects_dirty_source(tool):
    (tool / 'upstream/model.py').write_text('VALUE = 2\n')
    with pytest.raises(RuntimeError, match='tracked modifications'):
        activate(tool, ())


def test_rejects_wrong_pin(tool):
    (tool / 'upstream.json').write_text(json.dumps({'commit': '0' * 40}))
    with pytest.raises(RuntimeError, match='Wrong upstream commit'):
        activate(tool, ())


def test_rejects_foreign_editable_import(tool, monkeypatch):
    module = ModuleType('migration_fixture')
    module.__file__ = '/foreign/checkout/model.py'
    monkeypatch.setitem(sys.modules, 'migration_fixture', module)
    with pytest.raises(RuntimeError, match='already imported from another checkout'):
        activate(tool, ('migration_fixture',))


def test_rejects_foreign_namespace(tool, monkeypatch):
    module = ModuleType('migration_fixture')
    module.__path__ = ['/foreign/seed/modules']
    monkeypatch.setitem(sys.modules, 'migration_fixture', module)
    with pytest.raises(RuntimeError, match='package path from another checkout'):
        activate(tool, ('migration_fixture',))


def test_namespace_does_not_merge_legacy_editable_root(tool, monkeypatch):
    package = tool / 'upstream/migration_fixture'
    package.mkdir()
    foreign = tool / 'foreign'
    (foreign / 'migration_fixture').mkdir(parents=True)
    monkeypatch.syspath_prepend(str(foreign))
    monkeypatch.setitem(sys.modules, 'migration_fixture', None)
    del sys.modules['migration_fixture']
    activate(tool, ('migration_fixture',))
    module = sys.modules['migration_fixture']
    assert list(module.__path__) == [str(package)]
    # Adding another search path must not trigger namespace recombination.
    monkeypatch.syspath_prepend(str(foreign))
    assert list(module.__path__) == [str(package)]
