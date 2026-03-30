#!/bin/bash
# 编译 main.tex 并清理中间文件，只保留 PDF

TEX_FILE="main.tex"
PDF_FILE="${TEX_FILE%.tex}.pdf"
BASE_NAME="${TEX_FILE%.tex}"

cd "$(dirname "$0")" || exit 1

# 标准文献编译流程：pdflatex -> bibtex -> pdflatex -> pdflatex
pdflatex -interaction=nonstopmode "$TEX_FILE" && \
bibtex "$BASE_NAME" && \
pdflatex -interaction=nonstopmode "$TEX_FILE" && \
pdflatex -interaction=nonstopmode "$TEX_FILE"

if [ $? -eq 0 ] && [ -f "$PDF_FILE" ]; then
    echo "✅ 编译成功：$PDF_FILE"
    # 清理所有中间文件
    rm -f "${BASE_NAME}".{aux,log,out,toc,lof,lot,bbl,blg,nav,snm,vrb,fls,fdb_latexmk,synctex.gz}
    echo "🧹 中间文件已清理"
else
    echo "❌ 编译失败，保留日志文件以供排查"
    exit 1
fi
