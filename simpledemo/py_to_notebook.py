#!/usr/bin/env python3
"""
Script to convert Python files to Jupyter notebooks
"""

import json
import sys
import argparse
from pathlib import Path

def py_to_notebook(py_file, nb_file=None):
    """Convert Python file to Jupyter notebook"""
    
    if nb_file is None:
        nb_file = py_file.replace('.py', '.ipynb')
    
    # Read Python file
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by cells (using # %% as cell separator)
    if '# %%' in content:
        cells_content = content.split('# %%')[1:]  # Remove empty first element
    else:
        # If no cell markers, treat entire file as one cell
        cells_content = [content]
    
    # Create notebook structure
    cells = []
    for i, cell_content in enumerate(cells_content):
        cell_content = cell_content.strip()
        if cell_content:
            # Determine if it's markdown or code
            if cell_content.startswith('# markdown') or cell_content.startswith('"""'):
                cell_type = "markdown"
                # Remove markdown markers
                if cell_content.startswith('# markdown'):
                    cell_content = cell_content.replace('# markdown\n', '').strip()
                elif cell_content.startswith('"""') and cell_content.endswith('"""'):
                    cell_content = cell_content[3:-3].strip()
            else:
                cell_type = "code"
            
            cell = {
                "cell_type": cell_type,
                "metadata": {},
                "source": cell_content.split('\n')
            }
            
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            
            cells.append(cell)
    
    # Create notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (qlib)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Converted {py_file} → {nb_file}")
    return nb_file

def main():
    parser = argparse.ArgumentParser(description='Convert Python file to Jupyter notebook')
    parser.add_argument('py_file', help='Python file to convert')
    parser.add_argument('--output', '-o', help='Output notebook file (optional)')
    
    args = parser.parse_args()
    
    if not Path(args.py_file).exists():
        print(f"❌ Error: File {args.py_file} not found")
        sys.exit(1)
    
    nb_file = py_to_notebook(args.py_file, args.output)
    print(f"🎉 Notebook created: {nb_file}")
    print(f"💡 Open with: jupyter notebook {nb_file}")

if __name__ == "__main__":
    main()