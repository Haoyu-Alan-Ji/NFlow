from collections import OrderedDict

N = 5000
P = 200
N_ACTIVE = 10
SIGMA2 = 1.0
TARGET_SIGNAL_SD = 1.5
TRAIN_FRAC = 0.80
SEEDS = list(range(400, 410))

TEACHER_UNITS = 6
FEATURES_PER_UNIT = 2
N_INTERACTIONS = 2
N_QUADRATIC = 2
EXTRA_SCALE = 0.5
X_LOW = -2.5
X_HIGH = 2.5

HIDDEN_DIMS = (20, 20)
SELECTION_MODE = "feature_unit_induced_edge"
K_FLOW = 6
FLOW_HIDDEN_UNITS = 128
FLOW_HIDDEN_LAYERS = 2
SCALE_CLIP = 2.0
IAF_ORDERING = "cyclic3"
GATE_TYPE = "normalized_requ"
GATE_SCALE = 1.0
EPOCHS = 2000
WARMUP_EPOCHS = 300
LR = 3e-4
R_TRAIN = 32
R_EVAL = 128
R_FINAL = 500
INIT_SD = 0.5
INIT_LOC_JITTER = 0.05
GRAD_CLIP = 5.0
SUPPORT_THRESHOLD = 0.5
SLAB_INIT = "auto"
SLAB_SD_RATIO = 0.1
SLAB_BIAS_SD = 0.02

TABLE1_CONDITIONS = OrderedDict([
    ("relu", dict(activation="relu", n_interactions=0, n_quadratic=0)),
    ("relu_interaction", dict(activation="relu", n_interactions=N_INTERACTIONS, n_quadratic=0)),
    ("relu_quadratic", dict(activation="relu", n_interactions=0, n_quadratic=N_QUADRATIC)),
    ("trig", dict(activation="trig", n_interactions=0, n_quadratic=0)),
    ("trig_interaction", dict(activation="trig", n_interactions=N_INTERACTIONS, n_quadratic=0)),
])

TABLE1_LABELS = {
    "relu": "ReLU",
    "relu_interaction": "ReLU + interaction",
    "relu_quadratic": "ReLU + quadratic",
    "trig": "Trigonometric",
    "trig_interaction": "Trigonometric + interaction",
}

TABLE2_CONDITION = "trig_interaction"
TABLE2_METHODS = [
    "DSS-LVR",
    "LBBNN-LRT",
    "LBBNN-FLOW",
    "ISLaB-FLOW",
    "SS-GL",
    "SS-GHS",
    "IS-ANN-L1",
]
