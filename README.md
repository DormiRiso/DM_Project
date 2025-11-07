# DM_Project

## 🧩 Compilazione automatica con Makefile

Il progetto include un **Makefile** per gestire facilmente la compilazione del documento `main.tex`, la pulizia dei file temporanei e la generazione del PDF finale.

### ⚙️ Comandi principali

| Comando | Descrizione |
|----------|--------------|
| `make` | Compila `main.tex` e genera `main.pdf` |
| `make view` | Apre il PDF appena compilato |
| `make clean` | Rimuove i file temporanei di compilazione (`.aux`, `.log`, `.out`, ecc.) |
| `make cleanall` | Rimuove tutti i file di output, incluso `main.pdf` |

### 🔄 Ricompilazione automatica
Il Makefile ricompila automaticamente il PDF solo se:
- Cambi `main.tex`
- Modifichi un file `.tex` nella cartella `docs/`
- Aggiorni una figura (`.png`) in una sottocartella di `figures/`

### 🧹 Esempi rapidi

```bash
# Compila il documento
make

# Apre il PDF
make view

# Pulisce i file temporanei
make clean

# Elimina anche il PDF
make cleanall
