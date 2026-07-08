#!/bin/bash
set -e

echo "Generating paper figures..."
python -m src.figures.plot_paper_figures
