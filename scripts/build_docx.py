#!/usr/bin/env python3
"""Build a DOCX version of the paper by inlining tables and bibliography before pandoc."""
import os, re, subprocess, shutil, sys

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")
MAIN_TEX = os.path.join(PAPER_DIR, "main.tex")
OUT_DOCX = os.path.join(PAPER_DIR, "main.docx")
TMP_TEX = os.path.join(PAPER_DIR, "main_flat.tex")

def expand_inputs(tex_source: str) -> str:
    """Replace \\input{path} with the actual file contents."""
    def repl(m):
        path = m.group(1)
        if not path.endswith(".tex"):
            path = path + ".tex"
        full = os.path.join(PAPER_DIR, path)
        if not os.path.exists(full):
            print(f"WARNING: Could not find {full}", file=sys.stderr)
            return m.group(0)
        with open(full) as f:
            return f.read()
    return re.sub(r"\\input\{([^}]+)\}", repl, tex_source)

def inline_bibliography(tex_source: str) -> str:
    """Replace \\bibliography{...} with the contents of main.bbl (must be pre-built by pdflatex+bibtex)."""
    bbl = os.path.join(PAPER_DIR, "main.bbl")
    if not os.path.exists(bbl):
        print(f"WARNING: {bbl} not found. Run pdflatex+bibtex first.", file=sys.stderr)
        return tex_source
    with open(bbl) as f:
        bbl_content = f.read()
    # Remove bibliographystyle and bibliography lines, insert bbl content
    tex_source = re.sub(r"\\bibliographystyle\{[^}]+\}\s*", "", tex_source)
    # Use string replacement to avoid regex backreference interpretation of bbl_content
    tex_source = re.sub(r"\\bibliography\{[^}]+\}",
                        lambda m: bbl_content, tex_source)
    return tex_source

def main():
    # Ensure .bbl is up to date
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                   cwd=PAPER_DIR, capture_output=True)
    subprocess.run(["bibtex", "main"], cwd=PAPER_DIR, capture_output=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                   cwd=PAPER_DIR, capture_output=True)

    with open(MAIN_TEX) as f:
        src = f.read()

    src = expand_inputs(src)
    src = inline_bibliography(src)

    # Simplify some things pandoc 2.9 stumbles on
    # 1. The \blfootnote macro -> just regular \footnote (mark OK in docx)
    src = re.sub(r"\\newcommand\{\\blfootnote\}\[1\]\{%.*?\\endgroup\s*\}",
                 "", src, flags=re.DOTALL)
    src = src.replace("\\blfootnote{", "\\footnote{")

    # 2. Replace figures referenced as .pdf with .png (Word can't embed PDF inline)
    src = re.sub(r"figures/(\w+)\.pdf", r"figures/\1.png", src)
    # Also match the bare \includegraphics{figures/name} case (no extension)
    # → point to the .png explicitly
    src = re.sub(r"(\\includegraphics\[[^\]]*\])\{figures/(\w+)\}",
                 r"\1{figures/\2.png}", src)

    with open(TMP_TEX, "w") as f:
        f.write(src)

    # Convert with pandoc
    result = subprocess.run(
        ["pandoc", "main_flat.tex", "-o", "main.docx",
         "--mathml", "--standalone"],
        cwd=PAPER_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        print("pandoc stderr:", result.stderr, file=sys.stderr)
        sys.exit(1)
    print(result.stderr)

    # Clean up tmp
    if os.path.exists(TMP_TEX):
        os.remove(TMP_TEX)

    # Report
    if os.path.exists(OUT_DOCX):
        size = os.path.getsize(OUT_DOCX)
        print(f"Wrote {OUT_DOCX} ({size} bytes)")
    else:
        print("DOCX not produced", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
