"""Remplit le tableau 5.1 du rapport (latences de la version finale) à partir du journal d'AKORI.

AKORI enregistre, pour chaque question posée à l'assistant, le temps de recherche (BGE-M3 + FAISS),
le temps jusqu'au premier mot et le temps total, dans le fichier activity.log du dossier de données.

Utilisation (Colab, après avoir monté Google Drive) :
    python rapport/mesures_latence.py /content/drive/MyDrive/AKORI/activity.log
Sans argument, le script cherche activity.log dans les emplacements habituels.
"""
import json
import os
import statistics
import sys
from datetime import datetime

CANDIDATS = ["/content/drive/MyDrive/AKORI/activity.log", "/data/activity.log", "akori_data_store/activity.log"]
PROTOTYPE = {"first": (3.0, 6.0), "total": (10.0, 20.0)}   # prototype local Qwen (secondes), déjà dans le rapport


def centile(valeurs, p):
    v = sorted(valeurs)
    if not v:
        return None
    pos = (len(v) - 1) * p / 100
    bas, haut = int(pos), min(int(pos) + 1, len(v) - 1)
    return v[bas] + (v[haut] - v[bas]) * (pos - bas)


def fr(x, decimales=1):
    return f"{x:.{decimales}f}".replace(".", "{,}")


def main():
    chemin = sys.argv[1] if len(sys.argv) > 1 else next((c for c in CANDIDATS if os.path.isfile(c)), None)
    if not chemin or not os.path.isfile(chemin):
        sys.exit("Journal introuvable : indiquez le chemin de activity.log en argument.")
    mesures, dates, en_cache, refus = [], [], 0, 0
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            try:
                rec = json.loads(ligne)
            except ValueError:
                continue
            if rec.get("event") == "chat_cached":
                en_cache += 1
            elif rec.get("event") == "chat_refused":
                refus += 1
            m = rec.get("m") or {}
            if rec.get("event") == "chat" and m.get("total_ms"):
                mesures.append(m)
                dates.append(rec.get("t", "")[:10])
    if not mesures:
        sys.exit("Aucune question mesurée dans ce journal (événements « chat » avec des temps).")

    rech = [m["search_ms"] for m in mesures if m.get("search_ms") is not None]
    prem = [m["ttft_ms"] for m in mesures if m.get("ttft_ms") is not None]
    total = [m["total_ms"] / 1000 for m in mesures]
    med_r, med_p, med_t, p95_t = statistics.median(rech), statistics.median(prem), statistics.median(total), centile(total, 95)
    lo, hi = PROTOTYPE["total"]
    red_lo, red_hi = round((1 - med_t / lo) * 100), round((1 - med_t / hi) * 100)
    f_lo, f_hi = PROTOTYPE["first"][0] * 1000 / med_p, PROTOTYPE["first"][1] * 1000 / med_p
    jours = sorted(d for d in dates if d)
    debut = datetime.strptime(jours[0], "%Y-%m-%d").strftime("%d/%m/%Y")
    fin = datetime.strptime(jours[-1], "%Y-%m-%d").strftime("%d/%m/%Y")

    print(f"{len(mesures)} questions mesurées du {debut} au {fin} "
          f"({en_cache} réponses servies depuis le cache et {refus} questions hors cours, non comptées).\n")
    print(f"Recherche (BGE-M3 + FAISS) : médiane {med_r:.0f} ms")
    print(f"Premier mot affiché        : médiane {med_p:.0f} ms")
    print(f"Réponse complète           : médiane {med_t:.1f} s, 95e centile {p95_t:.1f} s\n")
    print("% --- lignes à coller dans le tableau 5.1 (rapport/main.tex) ---")
    print(f"Recherche FAISS & environ 50 ms & {med_r:.0f} ms (médiane) & Identique \\\\ \\hline")
    print(f"Premier mot affiché & 3 à 6 s & {med_p:.0f} ms (médiane) & {fr(f_lo)} à {fr(f_hi)} fois plus rapide \\\\ \\hline")
    print(f"Réponse complète & 10 à 20 s & {fr(med_t)} s (médiane), {fr(p95_t)} s (95\\textsuperscript{{e}} centile) "
          f"& $-${red_lo} à $-${red_hi}~\\% \\\\ \\hline")
    print(f"\nMesures réalisées sur {len(mesures)} questions, du {debut} au {fin}, avec \\aremplir{{environnement : Colab CPU/GPU ou serveur}}.")


if __name__ == "__main__":
    main()
