import ast
from pathlib import Path


def test_new_package_does_not_import_legacy_operations_logic():
    root = Path(__file__).resolve().parents[3] / "speech_edit/pipelines/text"
    assert root.is_dir()
    assert list(root.glob("*.py"))
    forbidden = {
        "EditOpGenClient",
        "EditWavGenClient",
        "f5_tts.infer.edit",
        "NodeGraph",
        "EditOperation",
    }
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden
                    assert not alias.name.startswith("f5_tts.infer.edit")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in forbidden
                assert not module.startswith("f5_tts.infer.edit")
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden
