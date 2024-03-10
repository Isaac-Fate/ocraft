#!/bin/sh

# Exit on error
set -e

# Create a new empty Python package directory
mkdir -p "$1"

# Create an empty __init__.py file
cd "$1"
touch __init__.py
