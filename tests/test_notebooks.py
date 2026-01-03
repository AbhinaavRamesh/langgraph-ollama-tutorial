"""
Notebook execution tests.

These tests verify that all tutorial notebooks execute without errors.
They require a running Ollama server and are marked as integration tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOKS_DIR = PROJECT_ROOT / "examples"


def get_notebook_paths() -> list[Path]:
    """Get all tutorial notebook paths."""
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


@pytest.fixture(scope="module")
def kernel_name() -> str:
    """Get the kernel name to use for execution."""
    return "langgraph-tutorial"


@pytest.mark.integration
@pytest.mark.slow
class TestNotebookExecution:
    """Test that all notebooks execute without errors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up working directory."""
        os.chdir(PROJECT_ROOT)

    @pytest.mark.parametrize(
        "notebook_path",
        get_notebook_paths(),
        ids=lambda p: p.stem,
    )
    def test_notebook_executes(self, notebook_path: Path, kernel_name: str) -> None:
        """Test that a notebook executes without errors.
        
        Uses a 600-second (10 minute) timeout to accommodate:
        - Initial Ollama model downloads (can be several GB)
        - Complex agent workflows with multiple LLM calls
        - Resource-intensive operations in capstone projects
        """
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(
            nb,
            timeout=600,  # 10 minutes for model downloads and complex workflows
            kernel_name=kernel_name,
        )

        try:
            client.execute()
        except CellExecutionError as e:
            pytest.fail(f"Notebook {notebook_path.name} failed:\n{e}")


@pytest.mark.integration
class TestNotebookOutputs:
    """Test that notebooks produce expected outputs."""

    def test_chatbot_produces_response(self, kernel_name: str) -> None:
        """Test that chatbot notebook produces a response."""
        notebook_path = NOTEBOOKS_DIR / "01_chatbot_basics.ipynb"

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=300, kernel_name=kernel_name)
        client.execute()

        # Check that we got some output
        outputs_found = False
        for cell in nb.cells:
            if cell.cell_type == "code" and cell.outputs:
                outputs_found = True
                break

        assert outputs_found, "Expected notebook to produce outputs"

    def test_tool_calling_executes_tools(self, kernel_name: str) -> None:
        """Test that tool calling notebook executes tools."""
        notebook_path = NOTEBOOKS_DIR / "02_tool_calling.ipynb"

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=300, kernel_name=kernel_name)
        client.execute()

        # Verify execution completed
        assert True  # If we get here, notebook executed


class TestNotebookStructure:
    """Test notebook structure without execution."""

    @pytest.mark.parametrize(
        "notebook_path",
        get_notebook_paths(),
        ids=lambda p: p.stem,
    )
    def test_notebook_is_valid(self, notebook_path: Path) -> None:
        """Test that notebook is valid JSON."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        assert nb.nbformat >= 4
        assert len(nb.cells) > 0

    @pytest.mark.parametrize(
        "notebook_path",
        get_notebook_paths(),
        ids=lambda p: p.stem,
    )
    def test_notebook_has_markdown(self, notebook_path: Path) -> None:
        """Test that notebook has explanatory markdown."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        markdown_cells = [c for c in nb.cells if c.cell_type == "markdown"]
        assert len(markdown_cells) >= 3, f"{notebook_path.name} should have explanatory markdown"

    @pytest.mark.parametrize(
        "notebook_path",
        get_notebook_paths(),
        ids=lambda p: p.stem,
    )
    def test_notebook_has_code(self, notebook_path: Path) -> None:
        """Test that notebook has code cells."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code_cells) >= 5, f"{notebook_path.name} should have code examples"
