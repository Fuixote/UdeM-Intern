#!/bin/bash
# Compile report0520.tex and clean intermediates, keeping only the PDF.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEX_FILE="surrogate_experiment_results/report0520.tex"
OUT_DIR="surrogate_experiment_results"
BASE_NAME="report0520"
PDF_FILE="${OUT_DIR}/${BASE_NAME}.pdf"

cd "$REPO_ROOT" || exit 1

# report0520.tex uses \RepoRoot=., so compile from the repository root.
# Output files are still written beside the .tex file.
status=0

pdflatex -interaction=nonstopmode -output-directory="$OUT_DIR" "$TEX_FILE" || status=$?

if [ "$status" -eq 0 ] && [ -f "${OUT_DIR}/${BASE_NAME}.aux" ] && grep -q '\\bibdata' "${OUT_DIR}/${BASE_NAME}.aux"; then
    bibtex "${OUT_DIR}/${BASE_NAME}" || status=$?
fi

if [ "$status" -eq 0 ]; then
    pdflatex -interaction=nonstopmode -output-directory="$OUT_DIR" "$TEX_FILE" || status=$?
fi

if [ "$status" -eq 0 ]; then
    pdflatex -interaction=nonstopmode -output-directory="$OUT_DIR" "$TEX_FILE" || status=$?
fi

if [ "$status" -eq 0 ] && [ -f "$PDF_FILE" ]; then
    echo "✅ 编译成功：$PDF_FILE"
    rm -f "${OUT_DIR}/${BASE_NAME}".{aux,log,out,toc,lof,lot,bbl,blg,nav,snm,vrb,fls,fdb_latexmk,synctex.gz}
    echo "🧹 中间文件已清理"
else
    echo "❌ 编译失败，保留日志文件以供排查"
    exit 1
fi
