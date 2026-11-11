"""Banc de mesure d'AKORI : recherche (BGE-M3 + FAISS), seuil de pertinence, indexation, OCR et réponses Gemini.

Il utilise les fonctions réelles d'akori.py (extraction PyMuPDF, découpage 800/80 par page, BGE-M3 normalisé,
FAISS, k = 4, seuil de similarité, pipeline de réponse complet). Le cours de test est un PDF passé en argument.

    python rapport/benchmark_akori.py cours.pdf [--gemini] [--ocr-pages 10] [--seulement ocr|gemini]

--seulement ne relance que la partie indiquée et complète les résultats déjà enregistrés.
Sorties : rapport/resultats_benchmark.json (toutes les mesures) et rapport/resultats_benchmark.tex
(macros LaTeX lues par main.tex : les tableaux du rapport se mettent à jour à la compilation).
"""
import json
import os
import statistics
import sys
import tempfile
import time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))

# --- questions : (question, mots attendus dans un passage ou une réponse correcte)
DANS_LE_COURS = [
    ("Quel modèle d'embedding utilise AKORI ?", ["bge-m3"]),
    ("Quelle bibliothèque assure la recherche vectorielle des passages ?", ["faiss"]),
    ("Quelle est la taille des fragments lors du découpage du texte ?", ["800"]),
    ("Quel chevauchement est conservé entre deux fragments successifs ?", ["80 caract"]),
    ("Combien de passages sont transmis au modèle génératif pour une question ?", ["k = 4", "quatre"]),
    ("Quel est le seuil de pertinence utilisé par défaut ?", ["0,30", "0.30"]),
    ("Quel modèle génératif principal est utilisé par AKORI ?", ["gemini-3.5-flash-lite", "flash-lite"]),
    ("Quelle méthode protège les mots de passe des comptes ?", ["pbkdf2"]),
    ("Combien d'itérations utilise la dérivation des mots de passe ?", ["200"]),
    ("Quelle bibliothèque extrait le texte des fichiers PDF ?", ["pymupdf"]),
    ("Quel moteur OCR lit les pages scannées ?", ["tesseract"]),
    ("Avec quel outil l'interface web a-t-elle été construite ?", ["gradio"]),
    ("Quel modèle de langage local a été testé au début du projet ?", ["qwen"]),
    ("Quelle latence observait le prototype exécuté localement ?", ["10 à 20", "10 et 20"]),
    ("Quels sont les intervalles de la répétition espacée du plan de révision ?", ["1, 3, 7", "14 jours"]),
    ("Quelle est la dimension des vecteurs produits par BGE-M3 ?", ["1024"]),
    ("Que signifie le sigle RAG ?", ["retrieval-augmented generation", "génération augmentée"]),
    ("Sur combien de points la colle orale est-elle notée ?", ["sur 10", "/10"]),
    ("Où sont enregistrées les données lorsque l'application tourne sur Colab ?", ["google drive"]),
]
REFORMULEES = [
    ("Quel outil compare les vecteurs pour retrouver les extraits les plus proches ?", ["faiss"]),
    ("Avec quel modèle les textes sont-ils transformés en listes de nombres ?", ["bge-m3"]),
    ("Comment l'application lit-elle les pages qui ne sont que des images ?", ["ocr", "tesseract"]),
    ("Que fait l'assistant quand aucun extrait n'est assez proche de la demande ?", ["ne figure pas", "seuil"]),
    ("Pourquoi a-t-on renoncé à faire tourner le modèle de langage sur la machine ?", ["latence", "10 à 20"]),
    ("Comment les secrets de connexion des utilisateurs sont-ils conservés de façon sûre ?", ["pbkdf2", "sel"]),
    ("Quelle est la longueur de chaque morceau de texte indexé ?", ["800"]),
    ("Sur quel principe de psychologie reposent les cartes mémoire et les questionnaires ?", ["effet de test", "récupération active"]),
    ("Comment l'application continue-t-elle de répondre quand le service d'IA est saturé ?", ["secours"]),
    ("Où peut-on héberger l'application pour qu'elle ait une adresse permanente ?", ["hugging face", "docker"]),
]
# Questions proches du sujet mais dont la réponse n'est pas dans le cours (vérifié dans le texte du PDF de test).
ABSENTES = [
    "Quelle est la taille maximale d'un PDF importé ?",
    "Combien coûte un abonnement mensuel à AKORI ?",
    "Quelle version de Python est utilisée dans l'image Docker ?",
    "Sur quel port réseau l'application écoute-t-elle sur le serveur ?",
    "Combien d'étudiants ont testé AKORI pendant le semestre ?",
    "AKORI fonctionne-t-il sur un Raspberry Pi ?",
]
HORS_SUJET = [
    "Quelle est la capitale de l'Australie ?", "Comment fonctionne la photosynthèse chez les plantes ?",
    "Qui a peint le tableau de la Joconde ?", "Quelle est la formule de l'aire d'un cercle ?",
    "Expliquez la théorie de la relativité restreinte d'Einstein.", "Comment préparer une pâte à crêpes ?",
    "Quelles sont les causes de la Première Guerre mondiale ?", "Quel est le rôle des mitochondries dans la cellule ?",
    "Comment calculer la dérivée d'une fonction polynomiale ?", "Qu'est-ce que le protocole de routage OSPF ?",
    "Quel est le plus long fleuve d'Afrique ?", "Comment fonctionne un moteur à explosion ?",
    "Quelles sont les règles du hors-jeu au football ?", "Qu'est-ce que l'inflation en économie ?",
    "Comment soigner une grippe saisonnière ?", "Quelle est la vitesse de la lumière dans le vide ?",
    "Qui a écrit le roman Les Misérables ?", "Comment fonctionne la blockchain du Bitcoin ?",
    "Quelles sont les étapes de la mitose ?", "Comment se forme un arc-en-ciel ?",
]


def series():
    return (("cours", DANS_LE_COURS), ("reformulee", REFORMULEES),
            ("absente", [(q, None) for q in ABSENTES]), ("hors_sujet", [(q, None) for q in HORS_SUJET]))


NOMS_SERIES = (("cours", "Cours"), ("reformulee", "Refo"), ("absente", "Abs"), ("hors_sujet", "Hors"))


def contient(texte, mots):
    t = str(texte).lower().replace(" ", " ").replace("\xa0", " ")
    return any(m in t for m in mots)


def med(v):
    return round(statistics.median(v), 1) if v else None


def centile(v, p):
    v = sorted(v)
    if not v:
        return None
    pos = (len(v) - 1) * p / 100
    b, h = int(pos), min(int(pos) + 1, len(v) - 1)
    return round(v[b] + (v[h] - v[b]) * (pos - b), 1)


def main():
    pdf = sys.argv[1]
    avec_gemini = "--gemini" in sys.argv
    ocr_pages = int(sys.argv[sys.argv.index("--ocr-pages") + 1]) if "--ocr-pages" in sys.argv else 0
    data = tempfile.mkdtemp(prefix="akori_bench_")
    os.environ.update(AKORI_DATA_DIR=data, AKORI_EMAIL_CHECK="0", AKORI_DRIVE_ROOT="/nonexistent")
    os.environ.setdefault("GEMINI_API_KEY", "AIzaBenchmarkPlaceholder")   # remplacée par la vraie clé côté proxy si besoin
    t0 = time.monotonic()
    import akori as A
    seul = sys.argv[sys.argv.index("--seulement") + 1] if "--seulement" in sys.argv else ""
    sortie = os.path.join(ICI, "resultats_benchmark.json")
    res = {"environnement": {}, "indexation": {}, "recherche": {}, "seuil": {}, "gemini": {}}
    if seul and os.path.exists(sortie):
        with open(sortie, encoding="utf-8") as f:
            res.update(json.load(f))
    if seul == "gemini":
        avec_gemini, ocr_pages = True, 0
    elif seul == "ocr":
        avec_gemini, ocr_pages = False, ocr_pages or 10
    ocr_prec = res["indexation"].get("ocr")
    env_prec, idx_prec = dict(res["environnement"]), dict(res["indexation"])
    res["environnement"] = {"date": time.strftime("%Y-%m-%d"), "cpu": os.cpu_count(), "gpu": A.EMBED_DEVICE, "modele_embedding": A.EMBED_MODEL_NAME,
                            "seuil": A.MIN_SIMILARITY, "k": A.RETRIEVAL_K, "chargement_s": round(time.monotonic() - t0, 1)}
    key = "bench@example.com"
    A._create_user(key, "benchmark123", onboarded=True, allow_username=True)
    A._CUR_USER.set(key)
    import fitz
    with fitz.open(pdf) as d:
        n_pages = len(d)

    def indexe(chemin):
        t = time.monotonic()
        msg = ""
        for ev in A.indexer_pdf_stream(chemin):
            if ev[0] == "done":
                msg = ev[2]
        return round(time.monotonic() - t, 1), msg

    duree, msg = indexe(pdf)
    nom = os.path.basename(pdf)
    res["indexation"] = {"pages": n_pages, "fragments": A.documents_db[nom]["chunks_count"], "duree_s": duree, "message": msg}
    if ocr_prec:
        res["indexation"]["ocr"] = ocr_prec
    print(f"Indexation : {n_pages} pages, {res['indexation']['fragments']} fragments en {duree} s")
    store = A.documents_db[nom]["vector_db"]

    def cherche(q):
        t = time.perf_counter()
        scored = store.similarity_search_with_score(q, k=A.RETRIEVAL_K)
        ms = (time.perf_counter() - t) * 1000
        sims = [max(0.0, min(1.0, 1.0 - float(dist) / 2.0)) for _, dist in scored]
        return ms, sims, [doc.page_content for doc, _ in scored], [(doc.metadata or {}).get("page") for doc, _ in scored]

    cherche("échauffement")
    if seul:
        lignes, temps = res.get("questions", []), []
    else:
        lignes, temps = [], []
    for serie, items in (() if seul else series()):
        for q, mots in items:
            ms, sims, textes, pages = cherche(q)
            temps.append(ms)
            trouve = None if mots is None else any(contient(t, mots) for t in textes)
            lignes.append({"serie": serie, "question": q, "sim_max": round(max(sims), 3), "ms": round(ms, 1),
                           "passage_pertinent": trouve, "pages": pages, "refus": max(sims) < A.MIN_SIMILARITY})
    res["questions"] = lignes
    if temps:
        res["recherche"] = {"n": len(temps), "mediane_ms": med(temps), "p95_ms": centile(temps, 95)}
    for serie in (() if seul else [c for c, _ in NOMS_SERIES]):
        sel = [l for l in lignes if l["serie"] == serie]
        res["seuil"][serie] = {"n": len(sel), "sim_min": min(l["sim_max"] for l in sel), "sim_max": max(l["sim_max"] for l in sel),
                               "sim_mediane": round(statistics.median(l["sim_max"] for l in sel), 3),
                               "refuses": sum(l["refus"] for l in sel),
                               "pertinent_dans_k": None if serie in ("hors_sujet", "absente") else sum(bool(l["passage_pertinent"]) for l in sel)}
    print(json.dumps(res["seuil"], ensure_ascii=False, indent=1))
    print("Recherche :", res["recherche"])

    if ocr_pages:   # même PDF, pages converties en images (sans couche de texte) : indexation avec OCR
        img_pdf = os.path.join(data, "scan_" + nom)
        with fitz.open(pdf) as d, fitz.open() as out:
            for i in range(min(ocr_pages, len(d))):
                pix = d[i].get_pixmap(dpi=150)   # page « scannée » : image JPEG seule, aucune couche de texte
                page = out.new_page(width=d[i].rect.width, height=d[i].rect.height)
                page.insert_image(page.rect, stream=pix.tobytes("jpeg", jpg_quality=80))
            out.save(img_pdf)
        with fitz.open(pdf) as d, fitz.open() as out:
            out.insert_pdf(d, to_page=min(ocr_pages, len(d)) - 1)
            txt_pdf = os.path.join(data, "texte_" + nom)
            out.save(txt_pdf)
        d_txt, _ = indexe(txt_pdf)
        d_ocr, m_ocr = indexe(img_pdf)
        res["indexation"]["ocr"] = {"pages": ocr_pages, "sans_ocr_s": d_txt, "avec_ocr_s": d_ocr, "moteur": A._ocr_label(), "message": m_ocr}
        print("OCR :", res["indexation"]["ocr"])

    if seul == "gemini" and env_prec:   # les mesures de recherche gardent leur environnement d'origine
        res["environnement"], env_gemini = env_prec, res["environnement"]
        res["indexation"] = idx_prec
    else:
        env_gemini = res["environnement"]

    if avec_gemini:
        rep = []
        for serie, items in series():
            for q, mots in items:
                A._ui_chat_user_message(q, [], nom)
                for _ in A.repondre_akori_chat_stream(q, nom):
                    pass
                item = A.documents_db[nom]["history"][-1]
                txt = str(item.get("content", ""))
                timing = item.get("timing") or {}
                rep.append({"serie": serie, "question": q, "reponse": txt[:1500], "timing": timing,
                            "refus": "ne figure pas" in txt.lower() or "not in the provided" in txt.lower(),
                            "secours": txt.startswith("ℹ️ L'IA est momentanément indisponible"),
                            "erreur": txt.startswith("⚠️"),
                            "correcte": None if mots is None else contient(txt, mots),
                            "citee": "[" in txt and "]" in txt, "sources": len(item.get("sources") or [])})
                time.sleep(1.0)
        ok = [r for r in rep if r["timing"].get("total")]
        res["gemini"] = {"n_mesurees": len(ok), "premier_mot_mediane_ms": med([r["timing"]["first"] for r in ok]),
                         "total_mediane_s": med([r["timing"]["total"] / 1000 for r in ok]),
                         "total_p95_s": centile([r["timing"]["total"] / 1000 for r in ok], 95),
                         "recherche_mediane_ms": med([r["timing"]["search"] for r in ok]),
                         "erreurs": sum(r["erreur"] or r["secours"] for r in rep), "environnement": env_gemini,
                         "date": time.strftime("%Y-%m-%d"), "reponses": rep}
        for serie, _ in NOMS_SERIES:
            sel = [r for r in rep if r["serie"] == serie]
            res["gemini"][serie] = {"n": len(sel), "correctes": sum(bool(r["correcte"]) for r in sel),
                                    "refus": sum(r["refus"] for r in sel), "citees": sum(r["citee"] for r in sel)}
        print(json.dumps({k: v for k, v in res["gemini"].items() if k != "reponses"}, ensure_ascii=False, indent=1))

    with open(sortie, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    ecrire_tex(res)


def _date_fr(iso):
    """2026-11-11 -> 11 novembre 2026."""
    mois = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()
    try:
        a, m, j = (int(x) for x in iso.split("-"))
        return f"{j} {mois[m - 1]} {a}"
    except ValueError:
        return iso


def _fr(x, dec=1):
    """Nombre au format français pour LaTeX (virgule décimale)."""
    if x is None:
        return r"\nonmesure"
    if isinstance(x, (int,)) or float(x).is_integer() and dec == 0:
        return str(int(round(x)))
    return f"{x:.{dec}f}".replace(".", "{,}")


def ecrire_tex(res):
    """Écrit rapport/resultats_benchmark.tex : une macro par valeur, lue par main.tex."""
    env, idx, rch, se, ge = res["environnement"], res["indexation"], res["recherche"], res["seuil"], res.get("gemini") or {}
    ocr = idx.get("ocr") or {}
    m = {
        "BenchDate": _date_fr(env.get("date", "")), "BenchCPU": str(env.get("cpu", "")),
        "BenchMateriel": "GPU" if env.get("gpu") not in (None, "cpu") else "processeur seul (sans GPU)",
        "BenchChargement": _fr(env.get("chargement_s")),
        "IdxPages": str(idx.get("pages", "")), "IdxFragments": str(idx.get("fragments", "")),
        "IdxDuree": _fr(idx.get("duree_s")),
        "IdxParPage": _fr(idx["duree_s"] / idx["pages"]) if idx.get("pages") else r"\nonmesure",
        "OcrPages": str(ocr.get("pages", "")) or r"\nonmesure",
        "OcrSans": _fr(ocr.get("sans_ocr_s")), "OcrAvec": _fr(ocr.get("avec_ocr_s") or None),
        "OcrParPage": _fr(ocr["avec_ocr_s"] / ocr["pages"]) if ocr.get("avec_ocr_s") else r"\nonmesure",
        "OcrSurcout": _fr((ocr["avec_ocr_s"] - ocr["sans_ocr_s"]) / ocr["pages"]) if ocr.get("avec_ocr_s") else r"\nonmesure",
        "RechN": str(rch.get("n", "")), "RechMed": _fr(rch.get("mediane_ms"), 0), "RechPq": _fr(rch.get("p95_ms"), 0),
    }
    for cle, nom in NOMS_SERIES:
        s = se.get(cle) or {}
        m[f"Seuil{nom}N"] = str(s.get("n", ""))
        m[f"Seuil{nom}Min"] = _fr(s.get("sim_min"), 2)
        m[f"Seuil{nom}Max"] = _fr(s.get("sim_max"), 2)
        m[f"Seuil{nom}Med"] = _fr(s.get("sim_mediane"), 2)
        m[f"Seuil{nom}Refus"] = str(s.get("refuses", ""))
        m[f"Seuil{nom}Acceptes"] = str(s["n"] - s["refuses"]) if s else ""
        if s.get("pertinent_dans_k") is not None:
            m[f"Seuil{nom}Rappel"] = str(s["pertinent_dans_k"])
            m[f"Seuil{nom}RappelPct"] = _fr(100 * s["pertinent_dans_k"] / s["n"], 0)
    g = {k: ge.get(k) for k in ("premier_mot_mediane_ms", "total_mediane_s", "total_p95_s")}
    m["GemN"] = str(ge.get("n_mesurees", 0))
    m["GemPremier"] = _fr(g["premier_mot_mediane_ms"], 0)
    m["GemTotal"] = _fr(g["total_mediane_s"])
    m["GemTotalPq"] = _fr(g["total_p95_s"])
    m["GemReduction"] = _fr(100 * (1 - g["total_mediane_s"] / 15), 0) if g["total_mediane_s"] else r"\nonmesure"
    for cle, nom in NOMS_SERIES:
        s = ge.get(cle) or {}
        for champ in ("correctes", "refus", "citees"):
            v = s.get(champ)
            m[f"Gem{nom}{champ.capitalize()}"] = str(v) if v is not None else r"\nonmesure"
            m[f"Gem{nom}{champ.capitalize()}Pct"] = _fr(100 * v / s["n"], 0) if v is not None and s.get("n") else r"\nonmesure"
    lignes = ["% Fichier généré par rapport/benchmark_akori.py : ne pas modifier à la main.",
              r"\providecommand{\nonmesure}{n.\,m.}"]
    lignes += [r"\newcommand{\%s}{%s}" % (k, v) for k, v in m.items()]
    with open(os.path.join(ICI, "resultats_benchmark.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    ecrire_tableau_questions(res)
    ecrire_balayage(res)


def ecrire_balayage(res):
    """Écrit rapport/resultats_balayage.tex : effet de différents seuils sur les questions du banc."""
    q = res.get("questions", [])
    dedans = [l["sim_max"] for l in q if l["serie"] in ("cours", "reformulee")]
    absentes = [l["sim_max"] for l in q if l["serie"] == "absente"]
    hors = [l["sim_max"] for l in q if l["serie"] == "hors_sujet"]
    out = [r"% Fichier généré par rapport/benchmark_akori.py : ne pas modifier à la main.",
           r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
           r"Seuil & \makecell{Questions du cours\\acceptées (sur %d)} & \makecell{Réponse absente\\refusées (sur %d)} & \makecell{Hors sujet\\refusées (sur %d)} \\" % (len(dedans), len(absentes), len(hors)),
           r"\midrule"]
    for k in range(30, 56, 5):
        t = k / 100
        ligne = r"%s%s & %d & %d & %d \\" % (_fr(t, 2), r" (défaut)" if k == 30 else "",
                                              sum(v >= t for v in dedans), sum(v < t for v in absentes), sum(v < t for v in hors))
        out.append(ligne)
    out += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(ICI, "resultats_balayage.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _tex(t):
    t = str(t)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        t = t.replace(a, b)
    return t


def ecrire_tableau_questions(res):
    """Écrit rapport/resultats_questions.tex : le détail question par question (annexe du rapport)."""
    noms = {"cours": "Dans le cours", "reformulee": "Reformulées", "absente": "Proches du sujet, réponse absente du cours", "hors_sujet": "Hors sujet"}
    rep = {(r["serie"], r["question"]): r for r in (res.get("gemini") or {}).get("reponses", [])}
    out = [r"% Fichier généré par rapport/benchmark_akori.py : ne pas modifier à la main.",
           r"\begin{longtable}{@{}r p{0.50\textwidth} c c c c@{}}",
           r"\caption{Résultat de chaque question du banc de test.}\label{tab:questions}\\",
           r"\toprule \# & Question & Sim. max & Passage & Seuil & Réponse \\ \midrule \endfirsthead",
           r"\toprule \# & Question & Sim. max & Passage & Seuil & Réponse \\ \midrule \endhead",
           r"\bottomrule \endfoot"]
    serie_prec = None
    for i, l in enumerate(res.get("questions", []), 1):
        if l["serie"] != serie_prec:
            if serie_prec is not None:
                out.append(r"\midrule")
            out.append(r"\multicolumn{6}{@{}l}{\textbf{%s}} \\" % noms[l["serie"]])
            serie_prec = l["serie"]
        passage = "--" if l["passage_pertinent"] is None else ("oui" if l["passage_pertinent"] else "non")
        seuil = "refus" if l["refus"] else "accepté"
        r = rep.get((l["serie"], l["question"]))
        if r is None:
            reponse = r"\nonmesure"
        elif l["serie"] in ("hors_sujet", "absente"):
            reponse = "refus" if r["refus"] else "répond"
        else:
            reponse = "refus" if r["refus"] else ("juste" if r["correcte"] else "à vérifier")
        out.append(r"%d & %s & %s & %s & %s & %s \\" % (i, _tex(l["question"]), _fr(l["sim_max"], 2), passage, seuil, reponse))
    out.append(r"\end{longtable}")
    with open(os.path.join(ICI, "resultats_questions.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    if "--tex" in sys.argv:   # régénère seulement les macros LaTeX à partir du JSON
        with open(os.path.join(ICI, "resultats_benchmark.json"), encoding="utf-8") as f:
            ecrire_tex(json.load(f))
    else:
        main()
