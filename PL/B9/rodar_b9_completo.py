from pathlib import Path
import subprocess
import sys


PASTA = Path(__file__).resolve().parent
ARQUIVOS = [
    "main_b9_mmgd.py",
    "plotar_b9_mmgd.py",
    "main_b9_mmgd_baterias.py",
    "plotar_b9_mmgd_baterias.py",
]


for arquivo in ARQUIVOS:
    print(f"\nExecutando {arquivo}...")
    subprocess.run([sys.executable, str(PASTA / arquivo)], check=True)

print("\nSimulações concluídas")
