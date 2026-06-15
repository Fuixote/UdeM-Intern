#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEX_FILE="intern01.tex"
BASE_NAME="${TEX_FILE%.tex}"
PDF_FILE="${BASE_NAME}.pdf"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "Error: pdflatex was not found in PATH." >&2
  exit 1
fi

cd "$SCRIPT_DIR"

if [[ ! -f "$TEX_FILE" ]]; then
  echo "Error: cannot find $SCRIPT_DIR/$TEX_FILE" >&2
  exit 1
fi

echo "Compiling $SCRIPT_DIR/$TEX_FILE"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" && \
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE"

if [[ $? -eq 0 && -f "$PDF_FILE" ]]; then
  echo "Compiled successfully: $SCRIPT_DIR/$PDF_FILE"
  rm -f "${BASE_NAME}".{aux,log,out,toc,lof,lot,bbl,blg,nav,snm,vrb,fls,fdb_latexmk,synctex.gz}
  echo "Cleaned LaTeX intermediate files"
else
  echo "Compilation failed; keeping log files for debugging" >&2
  exit 1
fi
