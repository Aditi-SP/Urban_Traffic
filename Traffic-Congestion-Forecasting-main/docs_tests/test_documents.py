"""
Consistency tests for the research paper (.docx) and the slides (.pptx). They read the saved files directly (zip + regex,
no extra packages) and compare them with the verified evaluation outputs.
These are NOT part of the core software suite (tests/): run them separately with
    python -m unittest discover -s docs_tests -v
They only READ the documents and skip cleanly if the paper, the slides or the evaluation outputs are not present.
If you edit the paper or slides yourself, a failure here is expected when the text no longer matches model/metrics.json.
"""
import hashlib
import html
import json
import re
import sys
import unittest
import zipfile
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

PAPER = root_dir / "doucments" / "Research Paper" / "Traffic_Congestion_Prediction_Research_Paper.docx"
SLIDES = root_dir / "doucments" / "ppt" / "Traffic-Flow-OptimizationLSTM-and-XGBoost.pptx"
BACKUP = root_dir / "doucments" / "original_backup"
METRICS = root_dir / "model" / "metrics.json"
M = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.is_file() else None
MINUS = "−"
FILES_PRESENT = PAPER.is_file() and SLIDES.is_file() and BACKUP.is_dir() and METRICS.is_file()


def _text(xml: str, para_tag: str, text_tag: str) -> str:
    paras = re.findall(rf"<{para_tag}[ >].*?</{para_tag}>", xml, flags=re.S)
    out = []
    for p in paras:
        out.append("".join(re.findall(rf"<{text_tag}[^>]*>(.*?)</{text_tag}>", p, flags=re.S)))
    return html.unescape("\n".join(out))


def paper_text() -> str:
    return _text(zipfile.ZipFile(PAPER).read("word/document.xml").decode("utf8"), "w:p", "w:t")


def slide_texts() -> list:
    z = zipfile.ZipFile(SLIDES)
    names = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                   key=lambda s: int(re.findall(r"\d+", s)[0]))
    return [_text(z.read(n).decode("utf8"), "a:p", "a:t") for n in names]


def fmt(x, k=2):
    return f"{x:.{k}f}".replace("-", MINUS)


FORBIDDEN = ["0.9995", "0.0287", "0.999496", "near-perfect", "Outperformed", "outperforms", "significantly higher performance",
             "Zheng", "Brownlee", "Xu et al", "Li et al", "max_depth=6", "64, 32 units", "Google Maps", "underfitting",
             "Bangalore Traffic Data Portal", "peak hours", "morning/evening", "Rush hour cycles"]


@unittest.skipUnless(FILES_PRESENT, "paper, slides, their backup folder or model/metrics.json not found")
class DocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paper = paper_text()
        cls.slides = slide_texts()
        cls.all = {"paper": cls.paper, "slides": "\n".join(cls.slides)}

    def test_documents_open_and_backups_are_preserved(self):
        for f in (PAPER, SLIDES):
            self.assertIsNone(zipfile.ZipFile(f).testzip(), f.name)
        for name in (PAPER.name, SLIDES.name):
            self.assertTrue((BACKUP / name).exists(), f"original of {name} must stay in doucments/original_backup")
        self.assertEqual(len(self.slides), 11)

    def test_no_leaky_metrics_or_unsupported_claims(self):
        for doc, text in self.all.items():
            for bad in FORBIDDEN:
                self.assertNotIn(bad, text, f"{doc} still contains {bad!r}")

    def test_rush_hour_and_real_time_only_appear_in_honest_contexts(self):
        for doc, text in self.all.items():
            for kw, ok in (("rush", ("no ", "not ", "cannot", "impossible", "without", "No ")),
                           ("real-time", ("unverified", "not ", "would", "TomTom", "Real-time provider")),
                           ("Real-time", ("unverified", "not ", "would", "TomTom", "Real-time provider")),
                           ("live", ("not ", "Live Prediction Demo", "live demo", "live service", "live traffic", "live forecasting", "live tab"))):
                for m in re.finditer(kw, text):
                    ctx = text[max(0, m.start() - 110): m.end() + 110]
                    self.assertTrue(any(o in ctx for o in ok), f"{doc}: unqualified {kw!r}: ...{ctx}...")

    def test_metrics_match_the_verified_evaluation(self):
        met, ci = M["metrics"], M["uncertainty"]["rmse_ci95"]
        for name in ("Persistence baseline", "Road-mean baseline", "LSTM", "XGBoost"):
            for needle in (fmt(met[name]["MAE"]), fmt(met[name]["RMSE"]), fmt(met[name]["R2"], 3),
                           f"[{ci[name][0]:.2f}, {ci[name][1]:.2f}]"):
                for doc in ("paper", "slides"):
                    self.assertIn(needle, self.all[doc], f"{doc} lacks {needle} ({name})")

    def test_test_set_description(self):
        for doc, text in self.all.items():
            for needle in ("856", "2024-03-21", "2024-08-09"):
                self.assertIn(needle, text, doc)

    def test_road_mean_is_stated_as_best_and_no_improvement_is_claimed(self):
        self.assertRegex(self.paper, r"road-mean baseline obtained the lowest")
        self.assertIn("neither trained model demonstrates a supported improvement", self.paper.replace("\n", " "))
        self.assertRegex(self.all["slides"], r"road mean is best")
        self.assertRegex(self.all["slides"], r"neither (trained )?model shows a supported improvement")

    def test_honest_labels_for_provenance_demo_spark_and_tomtom(self):
        paper = self.paper.replace("\n", " ")
        for needle in ("unverified/unknown", "licence", "not live traffic forecasting", "unverified interface to the TomTom",
                       "local mode on a single computer", "involves no streaming", "not been tested against the live service"):
            self.assertIn(needle, paper)
        slides = self.all["slides"]
        for needle in ("licence unverified", "not live traffic", "unverified", "local mode on one computer", "not streaming"):
            self.assertIn(needle, slides)

    def test_paper_reference_list_is_consistent_and_verified_entries_only(self):
        text = self.paper
        refs = re.findall(r"^\[(\d+)\] (.+)$", text, flags=re.M)
        self.assertEqual([r[0] for r in refs], [str(i) for i in range(1, 10)])
        body = text[: text.index("7. References")]
        groups = [g for g in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", body) if "0" not in g]  # skip value ranges such as [0, 100]
        cited = {int(n) for grp in groups for n in re.split(r",\s*", grp)}
        self.assertEqual(cited, set(range(1, 10)), "every reference must be cited and every citation must exist")
        # DOIs verified against Crossref during the update
        verified = {"10.1145/2939672.2939785", "10.1162/neco.1997.9.8.1735", "10.1016/j.trc.2015.03.014",
                    "10.1109/TITS.2014.2345663", "10.1049/iet-its.2016.0208", "10.32604/cmes.2020.011013", "10.1145/2934664"}
        found = set(re.findall(r"https://doi\.org/(\S+)", text))
        self.assertEqual(found, verified)
        self.assertIn("2017", dict(refs)["5"])   # Zhao et al. is 2017, not 2019
        self.assertIn("2015", dict(refs)["4"])   # Lv et al. is 2015, not 2018

    def test_slide_reference_list_matches_paper_dois(self):
        slide_dois = set(re.findall(r"doi:(\S+)", "\n".join(self.slides)))
        paper_dois = set(re.findall(r"https://doi\.org/(\S+)", self.paper))
        self.assertEqual(slide_dois, paper_dois)

    def test_paper_images_are_the_current_verified_figures(self):
        z = zipfile.ZipFile(PAPER)
        blobs = {hashlib.md5(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/")}
        expected = [root_dir / "reports" / "figures" / "fig1_rmse_ci.png",
                    root_dir / "reports" / "figures" / "fig2_actual_vs_predicted_road.png",
                    root_dir / "reports" / "figures" / "fig3_predicted_vs_actual.png",
                    root_dir / "model" / "xgboost_feature_importance.png",
                    root_dir / "model" / "xgboost_training_curve.png"]
        for f in expected:
            self.assertIn(hashlib.md5(f.read_bytes()).hexdigest(), blobs, f"{f.name} is not embedded in the paper")
        old = zipfile.ZipFile(BACKUP / PAPER.name)
        old_blobs = {hashlib.md5(old.read(n)).hexdigest() for n in old.namelist() if n.startswith("word/media/")}
        self.assertFalse(blobs & old_blobs, "old (leaky) figures must not remain in the paper")

    def test_template_and_authors_are_preserved(self):
        self.assertIn("Devashish Rawat, Akshit Shukla, Ayushmaan Kumar", self.paper)
        self.assertIn("Jaypee University Guna", self.paper)
        self.assertIn("Traffic Congestion Prediction using LSTM and XGBoost: A Comparative Analysis on Bengaluru Traffic Data", self.paper)
        for h in ("Abstract", "1. Introduction", "2. Related Work", "3. Dataset Description", "4. Methodology",
                  "5. Results and Discussion", "6. Conclusion", "7. References"):
            self.assertIn(h, self.paper)
        self.assertIn("By: Devashish Rawat", self.slides[0])


if __name__ == "__main__":
    unittest.main()
