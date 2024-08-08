#!/usr/bin/env bash


cd ..

if ! [[ -x "$(command -v pdoc3)" ]]; then
  echo 'Error: pdoc3 is not installed.' >&2
  exit 1
fi

set -o nounset
set -o errexit
set -o xtrace

rm -rf docs/api_docs
mkdir -p docs/api_docs
pdoc3 quantum_inferno --force --html --output-dir docs/api_docs -c show_type_annotations=True