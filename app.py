import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from itertools import islice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'decision_tree_modelB.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'metrics.json')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
DATASET_LOCAL = os.path.join(STATIC_DIR, 'heart.csv')
DATASET_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/heart.csv'

# Load model and metrics
model = joblib.load(MODEL_PATH)
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

# Ensure folders exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Feature order expected by the model
FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

app = Flask(__name__)


def get_plot_urls():
    exts = {'.png', '.jpg', '.jpeg'}
    if not os.path.isdir(PLOTS_DIR):
        return []
    exclude = {'roc_curve.png', 'feature_importance.png', 'confusion_matrix.png'}
    files = [
        f for f in os.listdir(PLOTS_DIR)
        if os.path.splitext(f)[1].lower() in exts and f not in exclude
    ]
    files.sort()
    return [url_for('static', filename=f'plots/{f}') for f in files]


def _load_eval_data():
    try:
        if os.path.exists(DATASET_LOCAL):
            df = pd.read_csv(DATASET_LOCAL)
        else:
            df = pd.read_csv(DATASET_URL)
        required = FEATURES + ['target']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'Kolom hilang pada dataset evaluasi: {missing}')
        print('[plots] menggunakan dataset evaluasi:', 'local' if os.path.exists(DATASET_LOCAL) else 'remote')
        return df
    except Exception as e:
        print('[plots] gagal memuat dataset evaluasi, pakai data sintetis. Error:', e)
        return _synthetic_data()


def _plot_feature_importance(model, X):
    fi = getattr(model, 'feature_importances_', None)
    if fi is None:
        return
    s = pd.Series(fi, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=s.values, y=s.index, palette='Blues')
    plt.title('Pentingnya Fitur')
    plt.xlabel('Nilai Kepentingan')
    plt.ylabel('Fitur')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred):
    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    plt.title('Matriks Kebingungan')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()


def _plot_roc(model, X_test, y_test):
    try:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title('Kurva ROC')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=150)
        plt.close()
    except Exception:
        pass


def generate_plots_if_missing():
    # Hanya buat metrics.json bila belum ada; tidak membuat file gambar apa pun
    if os.path.exists(METRICS_PATH):
        return
    try:
        df = _load_eval_data()
        X = df[FEATURES]
        y = df['target']
        # gunakan split agar metrik realistis
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        roc_auc = None
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            pass

        # Tidak membuat plot apapun sesuai permintaan

        with open(METRICS_PATH, 'w', encoding='utf-8') as f:
            json.dump({'accuracy': acc, 'roc_auc': roc_auc}, f, ensure_ascii=False, indent=2)
        metrics.update({'accuracy': acc, 'roc_auc': roc_auc})
    except Exception as e:
        print('[plots] gagal membuat grafik:', e)

def _synthetic_data(n=400, random_state=42):
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({
        'age': rng.randint(29, 78, size=n),
        'sex': rng.randint(0, 2, size=n),
        'cp': rng.randint(0, 4, size=n),
        'trestbps': rng.randint(90, 200, size=n),
        'chol': rng.randint(120, 564, size=n),
        'fbs': rng.randint(0, 2, size=n),
        'restecg': rng.randint(0, 2, size=n),
        'thalach': rng.randint(70, 210, size=n),
        'exang': rng.randint(0, 2, size=n),
        'oldpeak': rng.uniform(0.0, 6.5, size=n),
        'slope': rng.randint(0, 3, size=n),
        'ca': rng.randint(0, 4, size=n),
        'thal': rng.randint(0, 3, size=n),
    })
    logits = (
        0.03 * (df['age'] - 55) +
        0.02 * (df['trestbps'] - 130) +
        0.02 * (df['chol'] - 240) +
        0.5 * (df['cp'] == 0).astype(int) +
        0.4 * df['exang'] -
        0.03 * (df['thalach'] - 150) +
        0.2 * (df['oldpeak'])
    )
    probs = 1 / (1 + np.exp(-logits))
    df['target'] = (probs > 0.5).astype(int)
    print('[plots] memakai data sintetis untuk evaluasi')
    return df


def build_presets(model, max_each=5):
    try:
        df = _load_eval_data()
    except Exception:
        df = _synthetic_data()
    X = df[FEATURES]
    # predictions and confidence
    y_pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        proba1 = model.predict_proba(X)[:, 1]
    else:
        # fallback: 1.0 for predicted class
        proba1 = (y_pred == 1).astype(float)

    df_pred = X.copy()
    df_pred['pred'] = y_pred
    df_pred['proba'] = proba1

    risky = df_pred[df_pred['pred'] == 1].sort_values('proba', ascending=False)
    safe = df_pred[df_pred['pred'] == 0].sort_values('proba', ascending=True)

    presets = {}
    for i, (_, row) in enumerate(islice(risky.iterrows(), max_each), start=1):
        key = f'p{i}'
        entry = {feat: (round(float(row[feat]), 1) if feat == 'oldpeak' else int(row[feat])) for feat in FEATURES}
        entry['risk'] = 1
        presets[key] = entry
    for i, (_, row) in enumerate(islice(safe.iterrows(), max_each), start=1):
        key = f'n{i}'
        entry = {feat: (round(float(row[feat]), 1) if feat == 'oldpeak' else int(row[feat])) for feat in FEATURES}
        entry['risk'] = 0
        presets[key] = entry
    return presets


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    risk_prob = None
    is_risky = None

    # default values for convenience
    defaults = {
        'age': 57, 'sex': 1, 'cp': 0, 'trestbps': 130, 'chol': 250,
        'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
        'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2
    }

    form_values = {k: request.form.get(k, '') for k in FEATURES}

    if request.method == 'POST':
        try:
            # Convert inputs to correct dtypes
            x_input = []
            for feat in FEATURES:
                val = request.form.get(feat)
                if val is None or val == '':
                    raise ValueError(f'Missing value for {feat}')
                if feat in ['oldpeak']:
                    x_input.append(float(val))
                else:
                    x_input.append(int(float(val)))

            X = np.array([x_input])
            y_pred = model.predict(X)[0]
            is_risky = int(y_pred) == 1

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0][1]
                risk_prob = float(proba)

            if is_risky:
                prediction_text = 'Pasien Berisiko Penyakit Jantung'
            else:
                prediction_text = 'Pasien Sehat / Tidak Berisiko'

            # Keep submitted values in the form
            defaults.update({feat: request.form.get(feat) for feat in FEATURES})
        except Exception:
            prediction_text = 'Input tidak valid. Periksa kembali nilai fitur Anda.'

    generate_plots_if_missing()
    plot_urls = get_plot_urls()
    accuracy = metrics.get('accuracy')
    try:
        presets = build_presets(model)
    except Exception:
        presets = {}

    return render_template(
        'index.html',
        features=FEATURES,
        defaults=defaults,
        prediction_text=prediction_text,
        risk_prob=risk_prob,
        is_risky=is_risky,
        accuracy=accuracy,
        plot_urls=plot_urls,
        presets=presets,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
