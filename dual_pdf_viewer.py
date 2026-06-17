"""
Dual PDF Viewer — Synchronized Page Turner
==========================================
用法:
    python dual_pdf_viewer.py <slides.pdf> <notes.pdf>

操作:
    → / Space / PageDown   : 下一页（两窗口同步）
    ← / Backspace / PageUp : 上一页（两窗口同步）
    F                      : 切换当前窗口全屏
    Q / Escape             : 退出

依赖:
    pip install pymupdf pillow
"""

import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import fitz  # PyMuPDF


# ─────────────────────────────────────────────
#  PDF Renderer
# ─────────────────────────────────────────────

def render_page(doc: fitz.Document, page_index: int, max_width: int, max_height: int) -> ImageTk.PhotoImage:
    """Render a PDF page scaled to fit within (max_width, max_height)."""
    page = doc[page_index]
    rect = page.rect
    scale = min(max_width / rect.width, max_height / rect.height)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return ImageTk.PhotoImage(img)


# ─────────────────────────────────────────────
#  PDF Window
# ─────────────────────────────────────────────

class PDFWindow:
    def __init__(self, root: tk.Tk, doc: fitz.Document, title: str, is_main: bool = False):
        self.doc = doc
        self.total = len(doc)
        self.current = 0
        self.is_fullscreen = False
        self._photo = None  # keep reference

        if is_main:
            self.win = root
        else:
            self.win = tk.Toplevel(root)

        self.win.title(title)
        self.win.configure(bg="#1a1a1a")

        # Canvas for PDF rendering
        self.canvas = tk.Canvas(self.win, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom status bar
        self.status_bar = tk.Frame(self.win, bg="#2d2d2d", height=32)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_bar.pack_propagate(False)

        self.page_label = tk.Label(
            self.status_bar, text="", bg="#2d2d2d", fg="#aaaaaa",
            font=("Helvetica", 11)
        )
        self.page_label.pack(side=tk.LEFT, padx=12)

        title_label = tk.Label(
            self.status_bar, text=title, bg="#2d2d2d", fg="#666666",
            font=("Helvetica", 10)
        )
        title_label.pack(side=tk.RIGHT, padx=12)

        self._hide_cursor()
        self.win.bind("<Configure>", lambda e: self._on_resize(e))
        self.win.bind("<Enter>", lambda e: self._hide_cursor())
        self.win.bind("<Motion>", lambda e: self._hide_cursor())
        self.canvas.bind("<Enter>", lambda e: self._hide_cursor())
        self.canvas.bind("<Motion>", lambda e: self._hide_cursor())

    def _hide_cursor(self):
        self.win.config(cursor="none")
        self.canvas.config(cursor="none")
        self.status_bar.config(cursor="none")
        self.page_label.config(cursor="none")
        for child in self.status_bar.winfo_children():
            child.config(cursor="none")

    def show_page(self, index: int):
        self.current = max(0, min(index, self.total - 1))
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return
        photo = render_page(self.doc, self.current, w, h)
        self._photo = photo  # prevent GC
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=photo)
        self.page_label.config(text=f"Page  {self.current + 1}  /  {self.total}")

    def _on_resize(self, event):
        self.show_page(self.current)

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.win.attributes("-fullscreen", self.is_fullscreen)
        if self.is_fullscreen:
            self.status_bar.pack_forget()
        else:
            self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        # Fullscreen transitions can reset the cursor on some Tk backends,
        # so re-apply the hidden cursor for a short period after toggling.
        self._hide_cursor()
        self.win.after(0, self._hide_cursor)
        self.win.after(50, self._hide_cursor)
        self.win.after(200, self._hide_cursor)
        self.win.after(50, lambda: self.show_page(self.current))


# ─────────────────────────────────────────────
#  App Controller
# ─────────────────────────────────────────────

class DualPDFApp:
    def __init__(self, path_a: str, path_b: str):
        self.root = tk.Tk()
        self.root.geometry("900x600")

        doc_a = fitz.open(path_a)
        doc_b = fitz.open(path_b)

        if len(doc_a) != len(doc_b):
            answer = messagebox.askokcancel(
                "页数不同",
                f"两份 PDF 页数不同（{len(doc_a)} vs {len(doc_b)}）。\n继续运行，翻页到较少的那份结束时会停止。",
            )
            if not answer:
                sys.exit(0)

        self.total = min(len(doc_a), len(doc_b))
        self.current = 0

        # Create two windows
        self.win_a = PDFWindow(self.root, doc_a, "📽  幻灯片", is_main=True)
        self.win_b = PDFWindow(self.root, doc_b, "📝  讲稿")
        self.window_map = {
            self.root: self.win_a,
            self.win_b.win: self.win_b,
        }

        # Position windows side by side on startup
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"900x600+0+0")
        self.win_b.win.geometry(f"680x480+920+60")

        # Bind keys to root (main window)
        for widget in [self.root, self.win_b.win]:
            widget.bind("<Right>",    lambda e: self.next_page())
            widget.bind("<space>",    lambda e: self.next_page())
            widget.bind("<Next>",     lambda e: self.next_page())   # PageDown
            widget.bind("<Left>",     lambda e: self.prev_page())
            widget.bind("<BackSpace>",lambda e: self.prev_page())
            widget.bind("<Prior>",    lambda e: self.prev_page())   # PageUp
            widget.bind("<f>",        self._toggle_active_fullscreen)
            widget.bind("<F>",        self._toggle_active_fullscreen)
            widget.bind("<Escape>",   lambda e: self._quit())
            widget.bind("<q>",        lambda e: self._quit())

        self.win_b.win.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        # Initial render after window is drawn
        self.root.after(100, self._initial_render)

    def _initial_render(self):
        self.win_a.show_page(0)
        self.win_b.show_page(0)

    def next_page(self):
        if self.current < self.total - 1:
            self.current += 1
            self.win_a.show_page(self.current)
            self.win_b.show_page(self.current)

    def prev_page(self):
        if self.current > 0:
            self.current -= 1
            self.win_a.show_page(self.current)
            self.win_b.show_page(self.current)

    def _toggle_active_fullscreen(self, event):
        window = event.widget.winfo_toplevel()
        target = self.window_map.get(window, self.win_a)
        target.toggle_fullscreen()

    def _quit(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────
#  File Picker Fallback
# ─────────────────────────────────────────────

def pick_file(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=[("PDF files", "*.pdf")])
    root.destroy()
    if not path:
        print("已取消。")
        sys.exit(0)
    return path


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 3:
        path_a = sys.argv[1]
        path_b = sys.argv[2]
    else:
        print("未提供参数，请通过文件选择器选择两份 PDF。")
        path_a = pick_file("选择 幻灯片 PDF（左窗口）")
        path_b = pick_file("选择 讲稿 PDF（右窗口）")

    app = DualPDFApp(path_a, path_b)
    app.run()
