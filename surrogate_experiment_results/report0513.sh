#!/bin/bash
# 编译 main.tex 并清理中间文件，只保留 PDF

TEX_FILE="report0513.tex"
PDF_FILE="${TEX_FILE%.tex}.pdf"
BASE_NAME="${TEX_FILE%.tex}"

cd "$(dirname "$0")" || exit 1

# 编译流程：pdflatex -> optional bibtex -> pdflatex -> pdflatex
status=0

pdflatex -interaction=nonstopmode "$TEX_FILE" || status=$?

if [ "$status" -eq 0 ] && [ -f "${BASE_NAME}.aux" ] && grep -q '\\bibdata' "${BASE_NAME}.aux"; then
    bibtex "$BASE_NAME" || status=$?
fi

if [ "$status" -eq 0 ]; then
    pdflatex -interaction=nonstopmode "$TEX_FILE" || status=$?
fi

if [ "$status" -eq 0 ]; then
    pdflatex -interaction=nonstopmode "$TEX_FILE" || status=$?
fi

if [ "$status" -eq 0 ] && [ -f "$PDF_FILE" ]; then
    echo "✅ 编译成功：$PDF_FILE"
    # 清理所有中间文件
    rm -f "${BASE_NAME}".{aux,log,out,toc,lof,lot,bbl,blg,nav,snm,vrb,fls,fdb_latexmk,synctex.gz}
    echo "🧹 中间文件已清理"
else
    echo "❌ 编译失败，保留日志文件以供排查"
    exit 1
fi
