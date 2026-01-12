#!/bin/bash
# Build JupyterLite with tutorial notebooks
# Skip if pre-built notebooks already exist (for CI)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$WEBSITE_DIR")"
OUTPUT_DIR="$WEBSITE_DIR/public/notebooks"

# Check if pre-built notebooks already exist
if [ -f "$OUTPUT_DIR/index.html" ]; then
  echo "Pre-built JupyterLite notebooks found, skipping build."
  echo "To rebuild, delete $OUTPUT_DIR first."
  exit 0
fi

# Check if jupyter is available
if ! command -v jupyter &> /dev/null; then
  echo "Warning: jupyter not found. Skipping JupyterLite build."
  echo "Pre-built notebooks should be committed to the repository."
  exit 0
fi

set -e

echo "Building JupyterLite..."
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy notebooks to a temp directory for building
TEMP_NOTEBOOKS="$SCRIPT_DIR/content"
mkdir -p "$TEMP_NOTEBOOKS"

# Copy all tutorial notebooks
echo "Copying notebooks..."
cp -r "$PROJECT_ROOT/examples/core_patterns/"*.ipynb "$TEMP_NOTEBOOKS/" 2>/dev/null || true
cp -r "$PROJECT_ROOT/examples/rag_patterns/"*.ipynb "$TEMP_NOTEBOOKS/" 2>/dev/null || true
cp -r "$PROJECT_ROOT/examples/multi_agent_patterns/"*.ipynb "$TEMP_NOTEBOOKS/" 2>/dev/null || true
cp -r "$PROJECT_ROOT/examples/advanced_reasoning/"*.ipynb "$TEMP_NOTEBOOKS/" 2>/dev/null || true

# Build JupyterLite
cd "$SCRIPT_DIR"
jupyter lite build \
  --contents "$TEMP_NOTEBOOKS" \
  --output-dir "$OUTPUT_DIR" \
  --lite-dir "$SCRIPT_DIR"

# Cleanup
rm -rf "$TEMP_NOTEBOOKS"

echo "JupyterLite build complete!"
echo "Notebooks available at: $OUTPUT_DIR"
