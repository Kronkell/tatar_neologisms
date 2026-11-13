"""All parameters of the computational experiment in one place.

Experiment flow (main.py):
  for every feature informativeness rho, comparison budget m and random seed:
    1. generate a synthetic lexicon: G loanwords x K candidate replacements,
       each candidate has d linguistic features and a latent "nativeness" utility u
       (data.make_lexicon);
    2. simulate m*N/2 pairwise judgements of native speakers inside each
       loanword group under the Bradley-Terry model (data.sample_comparisons);
    3. fit three rankers (models.py) and measure within-group Kendall tau
       between the estimated and the true utilities (metrics.py);
  then aggregate seeds and compute how many comparisons plain BT needs to
  match the hierarchical model (metrics.equivalent_budget).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# --- synthetic lexicon -------------------------------------------------------
N_GROUPS = 40          # G: Russian loanwords to be replaced
N_CANDIDATES = 8       # K: Tatar candidate alternatives per loanword
N_FEATURES = 5         # d: linguistic features of a candidate
UTILITY_SCALE = 2.0    # std of the latent utility u (controls judgement noise)

# rho = Var(theta^T x) / Var(u): share of preference variance explained by features
RHOS = [0.1, 0.3, 0.5, 0.7, 0.9]

# m: average number of pairwise comparisons per candidate (budget)
BUDGETS = [0.5, 1, 2, 4, 8, 16, 32]

SEEDS = list(range(20))

# --- models ------------------------------------------------------------------
# Ridge strengths are selected on a 20% hold-out of comparisons from this grid,
# the same grid for plain BT and for the hierarchical model.
LAMBDA_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
HOLDOUT_FRACTION = 0.2
LAMBDA_THETA = 1e-2    # weak ridge on feature weights theta

METHODS = ["bt", "features", "hierarchical"]

# --- analysis ----------------------------------------------------------------
MAIN_RHO = 0.5           # rho shown in the left panel of the main figure
REFERENCE_BUDGETS = [1, 2, 4]   # m0 for the "budget saving" curves
N_BOOTSTRAP = 1000       # bootstrap over seeds for confidence intervals
BOOTSTRAP_SEED = 12345

N_JOBS = None            # worker processes (None = all CPU cores)

# =============================================================================
# Added after the first report (13.11). Original E1 parameters above are untouched.
# =============================================================================
RAW_DATA_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "tables"
LLM_DIR = ROOT / "data" / "llm"
V = UTILITY_SCALE ** 2   # Var(u) in the synthetic lexicon

# --- E1 additions (computed in the same run_one, written to results/e1_*.csv) ---
NEW_METHODS = ["bt_fixed", "hier_oracle"]   # fixed lambda: 1/V and 1/((1-rho)V)
N_CONTROL_SYNTH = 20000  # fresh control comparisons per seed (NLL on control, E6)

# --- E2: separate theory-check run (own RNG streams, independent of E1) ---------
THEORY_BUDGETS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
THEORY_SEEDS = list(range(30))
THEORY_METHODS = ["bt", "hierarchical", "bt_fixed", "hier_oracle"]
MC_GRID_MIN, MC_GRID_MAX, MC_GRID_N = 0.25, 512, 90   # log grid for expected_risk_mc
MC_DRAWS = 200           # realisations of the design in expected_risk_mc
MC_MODES = ["mean_field", "design", "full"]

# --- E3: survey designs ----------------------------------------------------------
DESIGNS = ["random", "cycle", "star"]
DESIGN_IDS = {"random": 0, "cycle": 1, "star": 2}
DESIGN_RHOS = [0.3, 0.5, 0.7]

# --- E4: real data (BenchLS + Word Complexity Lexicon) ---------------------------
RAW_FILES = {
    "BenchLS.txt": (
        "https://raw.githubusercontent.com/qiang2100/BERT-LS/"
        "3aff7312b2afb1fc5c9ab8953b939140b4817475/datasets/BenchLS.txt",
        "e0dbf8095d2dc4a9d4aed008d9e9a6a8b3975a3018cf2785f01f8e663b7dd629",
    ),
    "lexicon_annotations.tsv": (
        "https://raw.githubusercontent.com/mounicam/lexical_simplification/"
        "85788879c1a234e8cdf11b9425f3e65d8e3fe43e/word_complexity_lexicon/lexicon_annotations.tsv",
        "c40eaa01cc767fe0194d393bf755bafc8fa64e84e8fd5370951c21f459328e5c",
    ),
}
MIN_GROUP_SIZE = 4       # keep BenchLS groups with >= 4 WCL words
N_ANNOTATORS = 11
N_ANNOT_A = 6            # training annotators; the other 5 are control (B)
REAL_SEEDS = list(range(10))
REAL_BUDGETS = [0.25, 0.5, 1, 2, 4, 8, 16, 24]   # 24 fits into the smallest pool A (m <= 26)
REAL_M0S = [0.25, 0.5, 1, 2, 4]   # m0 for savings on real data
# feature sets of different strength (indices into REAL_FEATURES); 'all' is the E4 model
REAL_FEATURE_SETS = {"charlm": [3], "syllables": [2], "length": [1], "zipf": [0],
                     "zipf+length": [0, 1], "all": [0, 1, 2, 3]}
REAL_LAMBDA_SELECTION = "annotator"   # hold out all comparisons of one random A annotator
REAL_METHODS = ["bt", "features", "hierarchical"]
REAL_FEATURES = ["zipf", "length", "syllables", "charlm"]
N_CONTROL_REAL = 20000   # fixed random subsample of B comparisons per repeat
REAL_MC_DRAWS = 100      # subsamples of the pool for the 'full' analogue (split 0, every set)
CHARLM_K = 0.1           # add-k smoothing of the character trigram LM
CHARLM_TOPN = 50000      # training words: wordfreq.top_n_list("en", 50000)
EXPECTED_COUNTS = {"groups": 721, "words": 2641, "strict": 74477, "ties": 55656}

# --- E5 ------------------------------------------------------------------------
COMPLEXITY_BUDGETS = [2, 8]
LAMBDA_PATH = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
LAMBDA_PATH_BUDGET = 2

# --- E6 ------------------------------------------------------------------------
TABLE_SYNTH_RHOS = [0.3, 0.5, 0.7]
TABLE_SYNTH_BUDGET = 2
TABLE_REAL_BUDGETS = [2, 8]

# --- E7 ------------------------------------------------------------------------
RUNTIME_GROUPS = [10, 20, 40, 80, 160]
RUNTIME_BUDGET = 4
RUNTIME_REPEATS = 3
