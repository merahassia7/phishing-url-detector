
#  Phishing Detector v3 — XGBoost + Transformer + HTML
# type: ignore
import os, re, csv, time, unicodedata, warnings
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
import tldextract
import requests as req
from bs4 import BeautifulSoup

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from features import extract_features

warnings.filterwarnings('ignore')
TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# ── Flask ──
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# LOAD MODELS
print("Loading models...")

# XGBoost
xgb_model = joblib.load(os.path.join(BASE_DIR, 'models', 'phishing_model.pkl'))
feature_cols = joblib.load(os.path.join(BASE_DIR, 'models', 'xgb_feature_names.pkl'))
SECUREBERT_PHISHING_CLASS = 0
XGB_PHISHING_CLASS = 1
FINAL_THRESHOLD = 0.50
W_SECUREBERT = 0.45
W_XGBOOST = 0.35
W_ZERODAY = 0.20
W_HTML_BLEND = 0.29
W_NON_HTML_BLEND = 1.0 - W_HTML_BLEND

#Transformer files are stored in the project root next to app.py.
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'securebert_phishing')
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
hf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
hf_model.eval()

print(" All models loaded")

# WHITELIST
WHITELIST = {
    'google.com','youtube.com','gmail.com','github.com','stackoverflow.com',
    'linkedin.com','microsoft.com','amazon.com','wikipedia.org','paypal.com'
}

TRUSTED_EXACT_HOSTS = {
    'facebook.com', 'www.facebook.com',
    'instagram.com', 'www.instagram.com',
    'google.com', 'www.google.com',
    'youtube.com', 'www.youtube.com',
    'github.com', 'www.github.com',
}

WHITELIST_SUSPICIOUS_WORDS = {
    'login', 'signin', 'sign-in', 'verify', 'verification', 'update',
    'secure', 'security', 'account', 'unlock', 'confirm', 'password',
    'reset', 'auth', 'payment', 'wallet', 'invoice', 'bank', 'support',
    'ebayisapi'
}

KNOWN_BRANDS = [
    'paypal','apple','microsoft','amazon','facebook','instagram',
    'netflix','ebay','bank','whatsapp','google'
]
SHORTENERS = {'bit.ly', 'goo.gl', 'shorte.st', 'x.co', 'ow.ly', 't.co', 'tinyurl.com', 'rb.gy', 'cutt.ly'}
SUSPICIOUS_TLDS = {'ru', 'cn', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'top', 'xyz', 'club', 'online', 'site', 'click', 'link', 'live'}

def has_suspicious_whitelist_context(url):
    parsed_url = url if re.match(r'^[a-z][a-z0-9+.-]*://', url) else 'http://' + url
    parsed = urlparse(parsed_url)
    tail = f"{parsed.path} {parsed.query} {parsed.fragment}".lower()
    return any(word in tail for word in WHITELIST_SUSPICIOUS_WORDS)

def get_hostname(url):
    parsed_url = url if re.match(r'^[a-z][a-z0-9+.-]*://', url) else 'http://' + url
    return (urlparse(parsed_url).hostname or '').lower()

def estimate_zeroday_score(url, ext=None, features=None):
    """
    Full Zero-Day heuristic engine — 15 signals, returns (score 0-100, alerts list).
    score >= 70 → force MALICIOUS regardless of ML models.
    """
    if ext is None:
        ext = TLD_EXTRACTOR(url)

    score   = 0
    alerts  = []
    low_url = str(url).lower()
    domain  = ext.domain or ''
    subdomain = ext.subdomain or ''
    suffix  = ext.suffix or ''

    # ── 1. Malware keyword in URL ──────────────────────────────
    malware_kws = ['phishing','malware','hack','virus','trojan','ransomware',
                   'exploit','botnet','spyware','keylogger','darkweb','credential']
    for kw in malware_kws:
        if kw in low_url:
            alerts.append(f"🔴 Malicious keyword in URL: '{kw}'")
            score += 45; break

    # ── 2. Brand impersonation in domain ───────────────────────
    for brand in KNOWN_BRANDS:
        if brand in domain and domain != brand:
            alerts.append(f"🔴 Brand impersonation: '{brand}' in domain")
            score += 45; break

    # ── 3. Brand in subdomain ──────────────────────────────────
    if subdomain:
        for brand in KNOWN_BRANDS:
            if brand in subdomain:
                alerts.append(f"🔴 Brand '{brand}' used in subdomain")
                score += 40; break

    # ── 4. Suspicious TLD ─────────────────────────────────────
    if suffix in SUSPICIOUS_TLDS:
        alerts.append(f"⚠️ Suspicious TLD: .{suffix}")
        score += 20

    # ── 5. IP address as domain ───────────────────────────────
    if re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', domain):
        alerts.append("🔴 IP address used instead of domain name")
        score += 35

    # ── 6. Non-standard port ──────────────────────────────────
    if re.search(r':\d{2,5}/', low_url):
        alerts.append("⚠️ Non-standard port detected")
        score += 20

    # ── 7. URL-encoded obfuscation ────────────────────────────
    if re.search(r'%[0-9a-f]{2}', low_url):
        alerts.append("⚠️ URL-encoded characters (obfuscation attempt)")
        score += 15

    # ── 8. @ symbol in URL (credential hijack) ────────────────
    if '@' in low_url:
        alerts.append("🔴 @ symbol in URL — possible credential hijack")
        score += 40

    # ── 9. URL shortener ──────────────────────────────────────
    reg_domain = f"{domain}.{suffix}"
    if reg_domain in SHORTENERS:
        alerts.append("⚠️ URL shortener detected — real destination unknown")
        score += 20

    # ── 10. Multiple phishing keywords combo ──────────────────
    phish_kws = ['login','verify','secure','account','update','confirm',
                 'password','signin','credential','banking','payment','reset']
    kw_hits = [kw for kw in phish_kws if kw in low_url]
    if len(kw_hits) >= 3:
        alerts.append(f"🔴 Multiple phishing keywords ({len(kw_hits)}): {', '.join(kw_hits[:4])}")
        score += 35
    elif len(kw_hits) == 2:
        score += 15

    # ── 11. Brand subdomain + bad TLD combo ───────────────────
    if subdomain and suffix in SUSPICIOUS_TLDS:
        for brand in KNOWN_BRANDS:
            if brand in subdomain:
                alerts.append(f"🔴 Classic phishing pattern: '{brand}' subdomain + .{suffix}")
                score += 50; break

    # ── 12. Excessive hyphens in domain ───────────────────────
    hyphens = domain.count('-')
    if hyphens >= 3:
        alerts.append(f"⚠️ Excessive hyphens in domain ({hyphens})")
        score += 20
    elif hyphens == 2:
        score += 10

    # ── 13. Too many subdomains ────────────────────────────────
    sub_parts = [s for s in subdomain.split('.') if s]
    if len(sub_parts) >= 3:
        alerts.append("⚠️ Suspicious subdomain depth")
        score += 15

    # ── 14. Digit substitution (amaz0n, g00gle, paypa1) ───────
    if re.search(r'[a-z][0-9][a-z]|[a-z]{2}[0-9]', domain):
        alerts.append("🔴 Digit substitution in domain (e.g. amaz0n, g00gle)")
        score += 30

    # ── 15. HTTP (no HTTPS) ───────────────────────────────────
    if not low_url.startswith('https'):
        score += 10  # silent penalty

    return min(score, 100), alerts

def normalize_verdict(label):
    txt = str(label).strip().lower()
    if txt in {'benign', 'safe', 'legit', 'legitimate', 'good'}:
        return 'safe'
    if txt in {'malware', 'malicious', 'phishing', 'bad', 'fraud', 'scam'}:
        return 'phishing'
    return txt

# LOGGING
def log_result(url, label):
    log_path = os.path.join(BASE_DIR, 'datasets', 'collected_data.csv')
    write_header = not os.path.exists(log_path)
    verdict = normalize_verdict(label)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['url', 'label', 'timestamp'])
        w.writerow([url, verdict, time.strftime('%Y-%m-%d %H:%M:%S')])

# TRANSFORMER PREPROCESSING
def url_to_text(url: str) -> str:
    url = str(url).lower().strip()
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = unicodedata.normalize('NFKD', url).encode('ascii','ignore').decode('ascii')
    tokens = re.split(r'[^a-z0-9]', url)
    tokens = [t for t in tokens if t and len(t) > 1]
    return ' '.join(tokens[:50])

def predict_transformer(url: str):
    text = url_to_text(url)

    inputs = hf_tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    )

    with torch.no_grad():
        outputs = hf_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    # 🔥 FIXED LABEL MAPPING
    labels = {int(k): str(v).lower() for k, v in getattr(hf_model.config, 'id2label', {}).items()}
    phishing_idx = next(
        (idx for idx, label in labels.items() if any(word in label for word in ('phish', 'malicious', 'malware'))),
        SECUREBERT_PHISHING_CLASS
    )
    return probs[phishing_idx].item()

# HTML ANALYSIS
def analyze_html(url):
    score, reasons = 0.0, []

    try:
        r = req.get(url if url.startswith('http') else 'https://' + url, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')

        if soup.find('input', {'type': 'password'}):
            score += 0.3
            reasons.append("Password field detected")

        if len(soup.get_text()) < 100:
            score += 0.2
            reasons.append("Low content page")

    except:
        pass

    return score, reasons

# ROUTES
@app.route('/icon/<path:filename>')
def icon_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'icon'), filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        url = data.get('url', '').strip().lower()
        do_html = bool(data.get('analyze_html', True))

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # ── WHITELIST ──
        ext = TLD_EXTRACTOR(url)
        reg_domain = f"{ext.domain}.{ext.suffix}"
        host = get_hostname(url)

        if host in TRUSTED_EXACT_HOSTS and not has_suspicious_whitelist_context(url):
            log_result(url, 'benign')
            return jsonify({
                'prediction': 'benign',
                'confidence': 100,
                'xgb_score': 0,
                'transformer_score': 0,
                'html_score': 0,
                'reasons': ['Trusted exact host']
            })

        if reg_domain in WHITELIST and not has_suspicious_whitelist_context(url):
            log_result(url, 'benign')
            return jsonify({
                'prediction': 'benign',
                'confidence': 100,
                'xgb_score': 0,
                'transformer_score': 0,
                'html_score': 0,
                'reasons': ['Trusted domain']
            })

        # ── XGBOOST ──
        xgb_features = pd.DataFrame([extract_features(url)])
        xgb_features = xgb_features.reindex(columns=feature_cols, fill_value=0)
        xgb_phishing_idx = list(xgb_model.classes_).index(XGB_PHISHING_CLASS)
        xgb_score = float(xgb_model.predict_proba(xgb_features)[0][xgb_phishing_idx])

        # ── TRANSFORMER ──
        phishing_score = predict_transformer(url)

        # ── ZERO-DAY ENGINE (15 signals) ──
        zeroday_score_int, zd_alerts = estimate_zeroday_score(url, ext=ext)
        zeroday_score = zeroday_score_int / 100.0

        # ── HTML ──
        html_score_raw, html_reasons = (analyze_html(url) if do_html else (0.0, []))
        html_score = html_score_raw if isinstance(html_score_raw, float) and html_score_raw <= 1.0 else html_score_raw / 100.0

        # ── FINAL SCORE ──
        base_ensemble = (W_SECUREBERT * phishing_score) + (W_XGBOOST * xgb_score) + (W_ZERODAY * zeroday_score)
        final = (W_NON_HTML_BLEND * base_ensemble) + (W_HTML_BLEND * html_score) if do_html else base_ensemble

        # ── DECISION LOGIC ──
        # Zero-Day >= 70 always forces MALICIOUS
        # SecureBERT >= 85% alone forces MALICIOUS
        if zeroday_score_int >= 70:
            prediction = 'malware'
        elif phishing_score >= 0.85:
            prediction = 'malware'
        elif do_html and html_score >= 0.55:
            prediction = 'malware'
        elif final >= FINAL_THRESHOLD:
            prediction = 'malware'
        else:
            prediction = 'benign'

        all_alerts = zd_alerts + html_reasons

        result = {
            'prediction': prediction,
            'confidence': round(final*100, 2),
            'xgb_score': round(xgb_score*100, 2),
            'transformer_score': round(phishing_score*100, 2),
            'zeroday_score': round(zeroday_score_int, 2),
            'html_score': round(html_score*100, 2),
            'reasons': all_alerts,
            'source': f"AI Ensemble (XGBoost + SecureBERT + Zero-Day{' + HTML' if do_html else ''})"
        }
        log_result(url, prediction)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json() or {}
        urls = data.get('urls', [])
        if isinstance(urls, str):
            urls = [u.strip() for u in urls.splitlines() if u.strip()]
        if not isinstance(urls, list) or len(urls) == 0:
            return jsonify({'error': 'No URLs provided'}), 400

        do_html = bool(data.get('analyze_html', False))
        limited_urls = [str(u).strip().lower() for u in urls if str(u).strip()][:500]
        results = []

        for idx, url in enumerate(limited_urls, start=1):
            try:
                ext = TLD_EXTRACTOR(url)
                reg_domain = f"{ext.domain}.{ext.suffix}"
                host = get_hostname(url)

                if host in TRUSTED_EXACT_HOSTS and not has_suspicious_whitelist_context(url):
                    entry = {
                        'row': idx,
                        'url': url,
                        'prediction': 'benign',
                        'confidence': 100,
                        'xgb_score': 0,
                        'transformer_score': 0,
                        'html_score': 0
                    }
                    log_result(url, 'benign')
                    results.append(entry)
                    continue

                if reg_domain in WHITELIST and not has_suspicious_whitelist_context(url):
                    entry = {
                        'row': idx,
                        'url': url,
                        'prediction': 'benign',
                        'confidence': 100,
                        'xgb_score': 0,
                        'transformer_score': 0,
                        'html_score': 0
                    }
                    log_result(url, 'benign')
                    results.append(entry)
                    continue

                xgb_features = pd.DataFrame([extract_features(url)])
                xgb_features = xgb_features.reindex(columns=feature_cols, fill_value=0)
                xgb_phishing_idx = list(xgb_model.classes_).index(XGB_PHISHING_CLASS)
                xgb_score = float(xgb_model.predict_proba(xgb_features)[0][xgb_phishing_idx])

                phishing_score = predict_transformer(url)
                zeroday_score_int, _zd_alerts = estimate_zeroday_score(url, ext=ext)
                zeroday_score = zeroday_score_int / 100.0
                html_score_raw, _reasons = (analyze_html(url) if do_html else (0.0, []))
                html_score = html_score_raw if isinstance(html_score_raw, float) and html_score_raw <= 1.0 else html_score_raw / 100.0
                base_ensemble = (W_SECUREBERT * phishing_score) + (W_XGBOOST * xgb_score) + (W_ZERODAY * zeroday_score)
                final = (W_NON_HTML_BLEND * base_ensemble) + (W_HTML_BLEND * html_score) if do_html else base_ensemble

                if zeroday_score_int >= 70:
                    prediction = 'malware'
                elif phishing_score >= 0.85:
                    prediction = 'malware'
                elif final >= FINAL_THRESHOLD:
                    prediction = 'malware'
                else:
                    prediction = 'benign'
                log_result(url, prediction)
                results.append({
                    'row': idx,
                    'url': url,
                    'prediction': prediction,
                    'confidence': round(final*100, 2),
                    'xgb_score': round(xgb_score*100, 2),
                    'transformer_score': round(phishing_score*100, 2),
                    'zeroday_score': round(zeroday_score*100, 2),
                    'html_score': round(html_score*100, 2)
                })
            except Exception as row_error:
                results.append({'row': idx, 'url': url, 'error': str(row_error)})

        return jsonify({'count': len(results), 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def history():
    try:
        limit = request.args.get('limit', default=200, type=int)
        limit = min(max(limit, 1), 1000)
        log_path = os.path.join(BASE_DIR, 'datasets', 'collected_data.csv')
        if not os.path.exists(log_path):
            return jsonify({'count': 0, 'items': []})

        df = pd.read_csv(log_path)
        required_cols = {'url', 'label', 'timestamp'}
        if not required_cols.issubset(set(df.columns)):
            return jsonify({'count': 0, 'items': []})

        df = df[['url', 'label', 'timestamp']].dropna()
        if len(df) == 0:
            return jsonify({'count': 0, 'items': []})
        df = df.iloc[::-1].head(limit)

        items = []
        for _, row in df.iterrows():
            verdict = normalize_verdict(row['label'])
            items.append({
                'url': str(row['url']),
                'prediction': verdict,
                'timestamp': str(row['timestamp'])
            })

        return jsonify({'count': len(items), 'items': items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    try:
        log_path = os.path.join(BASE_DIR, 'datasets', 'collected_data.csv')
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['url', 'label', 'timestamp'])
        return jsonify({'ok': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
