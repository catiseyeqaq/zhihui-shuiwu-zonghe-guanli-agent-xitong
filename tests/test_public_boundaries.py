import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicBoundaryTests(unittest.TestCase):
    def test_public_tree_has_no_private_markers(self):
        # Genuine private-leak markers only. The repo's own domain vocabulary
        # (e.g. the `data_sources` module, real-town names, the public ky10
        # EPANET benchmark) is intentionally NOT flagged here, so this guard
        # catches secret/server leaks without breaking legitimate code.
        forbidden = (
            "/workdata",
            "172.30.",
            "root@",
            "webui.db",
            ".webui_secret_key",
        )
        for path in ROOT.rglob("*"):
            if (
                not path.is_file()
                or path == Path(__file__).resolve()
                or ".git" in path.parts
                or "__pycache__" in path.parts
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            self.assertFalse(any(marker.lower() in text for marker in forbidden), path)


    def test_required_public_entrypoints_exist(self):
        required = (
            ROOT / "README.md",
            ROOT / "requirements.txt",
            ROOT / "agent" / "water_management_agent.py",
            ROOT / "integrations" / "openwebui" / "water_agent_tool.py",
            ROOT / "water_resilience" / "src" / "pipeline.py",
        )
        self.assertTrue(all(path.exists() for path in required))
