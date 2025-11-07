# ============================================================
#  Makefile per progetto LaTeX: Data Mining Project
# ============================================================

# Nome principale del file TeX (senza estensione)
MAIN = main

# Comando di compilazione: puoi cambiare in pdflatex, xelatex o lualatex
LATEX = latexmk
LATEXFLAGS = -pdf -interaction=nonstopmode -shell-escape

# File e directory di output
OUTPUT = $(MAIN).pdf

# Lista dei file sorgenti (il Makefile ricompila se uno di questi cambia)
SRC = $(MAIN).tex $(wildcard docs/*.tex) $(wildcard figures/**/*.png)

# ============================================================
#  Regole principali
# ============================================================

.PHONY: all clean cleanall view

# Regola principale: compila il PDF
all: $(OUTPUT)

# Compilazione automatica con latexmk
$(OUTPUT): $(SRC)
	@echo "📘 Compilazione del file TeX..."
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "✅ Compilazione completata: $(OUTPUT)"

# Mostra il PDF (solo se esiste)
view: $(OUTPUT)
	@echo "👁️  Apro il PDF..."
	@xdg-open $(OUTPUT) >/dev/null 2>&1 &

# ============================================================
#  Pulizia
# ============================================================

# Pulisce i file temporanei della compilazione
clean:
	@echo "🧹 Rimozione dei file temporanei..."
	@latexmk -c
	@rm -f $(MAIN).bbl $(MAIN).run.xml $(MAIN).synctex.gz
	@echo "✅ Pulizia completata."

# Pulisce TUTTO (incluso il PDF)
cleanall: clean
	@echo "🔥 Rimozione completa del PDF..."
	@rm -f $(OUTPUT)
	@echo "✅ Tutti i file di output rimossi."
