"""
CLI interface for langgraph-ollama-local.

This module provides command-line utilities for running examples,
checking Ollama connectivity, and managing configurations.

Usage:
    langgraph-local check          Check Ollama connection
    langgraph-local list           List available examples
    langgraph-local run <example>  Run a specific example
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn


def check_connection() -> int:
    """Check connection to Ollama server."""
    from langgraph_ollama_local.config import LocalAgentConfig

    try:
        import httpx
    except ImportError:
        print("Error: httpx is required. Install with: pip install httpx")
        return 1

    config = LocalAgentConfig()
    print(f"Checking connection to {config.ollama.base_url}...")

    try:
        response = httpx.get(
            f"{config.ollama.base_url}/api/tags",
            timeout=config.ollama.timeout,
        )
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"Connection successful!")
            print(f"Available models: {len(models)}")
            for model in models[:5]:
                name = model.get("name", "unknown")
                size = model.get("size", 0) / (1024 * 1024 * 1024)
                print(f"  - {name} ({size:.1f} GB)")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
            return 0
        else:
            print(f"Connection failed: HTTP {response.status_code}")
            return 1
    except httpx.ConnectError:
        print(f"Connection failed: Could not connect to {config.ollama.base_url}")
        print("Is Ollama running? Start with: ollama serve")
        return 1
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1


def list_examples() -> int:
    """List available examples."""
    from pathlib import Path

    examples_dir = Path(__file__).parent.parent / "examples"

    if not examples_dir.exists():
        print("No examples directory found.")
        return 1

    notebooks = list(examples_dir.glob("*.ipynb"))
    scripts = list(examples_dir.glob("*.py"))

    if not notebooks and not scripts:
        print("No examples found yet. Examples will be added in future phases.")
        return 0

    print("Available examples:\n")

    if notebooks:
        print("Jupyter Notebooks:")
        for nb in sorted(notebooks):
            print(f"  - {nb.stem}")

    if scripts:
        print("\nPython Scripts:")
        for script in sorted(scripts):
            print(f"  - {script.stem}")

    return 0


def show_config() -> int:
    """Show current configuration."""
    from langgraph_ollama_local.config import LocalAgentConfig

    config = LocalAgentConfig()

    print("Current Configuration:")
    print("-" * 40)
    print("Ollama Settings:")
    print(f"  Host:        {config.ollama.host}")
    print(f"  Port:        {config.ollama.port}")
    print(f"  Model:       {config.ollama.model}")
    print(f"  Timeout:     {config.ollama.timeout}s")
    print(f"  Max Retries: {config.ollama.max_retries}")
    print(f"  Temperature: {config.ollama.temperature}")
    print(f"  Context:     {config.ollama.num_ctx}")
    print(f"  Base URL:    {config.ollama.base_url}")
    print()
    print("LangGraph Settings:")
    print(f"  Recursion Limit: {config.langgraph.recursion_limit}")
    print(f"  Checkpoint Dir:  {config.langgraph.checkpoint_dir}")
    print(f"  Streaming:       {config.langgraph.enable_streaming}")

    return 0


def main() -> NoReturn:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="langgraph-local",
        description="LangGraph Ollama Local - CLI tools for local agent development",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    subparsers.add_parser("check", help="Check Ollama connection")

    # List command
    subparsers.add_parser("list", help="List available examples")

    # Config command
    subparsers.add_parser("config", help="Show current configuration")

    # Run command (placeholder)
    run_parser = subparsers.add_parser("run", help="Run an example")
    run_parser.add_argument("example", help="Name of the example to run")
    run_parser.add_argument(
        "--model",
        default=None,
        help="Override the model to use",
    )

    args = parser.parse_args()

    if args.command == "check":
        sys.exit(check_connection())
    elif args.command == "list":
        sys.exit(list_examples())
    elif args.command == "config":
        sys.exit(show_config())
    elif args.command == "run":
        print(f"Running example: {args.example}")
        print("Note: Example runner will be implemented in Phase 7")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
