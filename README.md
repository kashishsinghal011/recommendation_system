<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Amazon Recommender System</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --navy:   #0A0E1A;
    --navy2:  #0F1629;
    --blue:   #2563EB;
    --cyan:   #00D4FF;
    --cyan2:  #00A8CC;
    --white:  #F0F4FF;
    --slate:  #8892A4;
    --card:   #111827;
    --border: rgba(0,212,255,0.15);
    --glow:   0 0 30px rgba(0,212,255,0.25);
  }

  html { scroll-behavior: smooth; }

  body {
    background: var(--navy);
    color: var(--white);
    font-family: 'Inter', sans-serif;
    overflow-x: hidden;
    line-height: 1.6;
  }

  /* ── Canvas bg ── */
  #starfield {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0; pointer-events: none;
  }

  /* ── Sections ── */
  section, header, footer { position: relative; z-index: 1; }

  /* ── HERO ── */
  header {
    min-height: 100vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center;
    padding: 4rem 2rem;
    border-bottom: 1px solid var(--border);
  }

  .badge {
    display: inline-block;
    background: rgba(37,99,235,0.2);
    border: 1px solid rgba(37,99,235,0.5);
    color: var(--cyan);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    padding: 0.4rem 1rem;
    border-radius: 2rem;
    margin-bottom: 2rem;
    animation: fadeDown 0.8s ease both;
  }

  h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2.2rem, 6vw, 5rem);
    font-weight: 900;
    line-height: 1.1;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--white) 0%, var(--cyan) 60%, var(--blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeDown 0.9s ease 0.15s both;
  }

  .hero-sub {
    max-width: 600px;
    margin: 1.5rem auto 2.5rem;
    color: var(--slate);
    font-size: 1.1rem;
    font-weight: 300;
    animation: fadeDown 1s ease 0.3s both;
  }

  /* Typewriter */
  .typewriter {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    color: var(--cyan);
    border-right: 2px solid var(--cyan);
    white-space: nowrap;
    overflow: hidden;
    animation: typing 3.5s steps(50,end) 1s both, blink 0.75s step-end infinite;
    max-width: 100%;
  }

  @keyframes typing { from { width: 0 } to { width: 100% } }
  @keyframes blink { 50% { border-color: transparent } }

  /* Pill badges */
  .pill-row {
    display: flex; flex-wrap: wrap; gap: 0.6rem;
    justify-content: center;
    margin-top: 2.5rem;
    animation: fadeUp 1s ease 0.5s both;
  }
  .pill {
    background: rgba(0,212,255,0.08);
    border: 1px solid var(--border);
    border-radius: 2rem;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    color: var(--cyan);
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Neural graph demo ── */
  .neural-wrap {
    width: 100%; max-width: 700px;
    margin: 3rem auto 0;
    animation: fadeUp 1s ease 0.7s both;
  }
  #neural-canvas { width: 100%; border-radius: 1rem; display: block; }

  /* ── Section layout ── */
  .section {
    padding: 5rem 2rem;
    max-width: 1000px;
    margin: 0 auto;
  }

  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    opacity: 0; transform: translateY(20px);
    transition: opacity 0.6s ease, transform 0.6s ease;
  }
  .section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.6rem, 3vw, 2.4rem);
    font-weight: 700;
    margin-bottom: 1rem;
    opacity: 0; transform: translateY(20px);
    transition: opacity 0.6s ease 0.1s, transform 0.6s ease 0.1s;
  }
  .section-desc {
    color: var(--slate);
    max-width: 640px;
    margin-bottom: 3rem;
    opacity: 0; transform: translateY(20px);
    transition: opacity 0.6s ease 0.2s, transform 0.6s ease 0.2s;
  }
  .reveal .section-label,
  .reveal .section-title,
  .reveal .section-desc { opacity: 1; transform: none; }

  /* ── Algo cards ── */
  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.25rem;
  }
  .algo-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 1.75rem;
    cursor: default;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    opacity: 0; transform: translateY(30px);
  }
  .algo-card.reveal { opacity: 1; transform: none; }
  .algo-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--glow);
    border-color: rgba(0,212,255,0.4);
  }
  .card-icon {
    font-size: 2rem; margin-bottom: 1rem;
  }
  .card-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--cyan);
    margin-bottom: 0.5rem;
    letter-spacing: 0.05em;
  }
  .card-body { color: var(--slate); font-size: 0.9rem; line-height: 1.6; }

  /* ── Pipeline ── */
  .pipeline {
    display: flex; flex-direction: column; gap: 0;
  }
  .pipe-step {
    display: flex; gap: 1.5rem; align-items: flex-start;
    opacity: 0; transform: translateX(-20px);
    transition: opacity 0.5s ease, transform 0.5s ease;
  }
  .pipe-step.reveal { opacity: 1; transform: none; }
  .pipe-left {
    display: flex; flex-direction: column; align-items: center;
    min-width: 48px;
  }
  .pipe-num {
    width: 48px; height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--blue), var(--cyan));
    display: flex; align-items: center; justify-content: center;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem; font-weight: 700;
    flex-shrink: 0;
  }
  .pipe-line {
    width: 2px; flex: 1; min-height: 2rem;
    background: linear-gradient(to bottom, var(--cyan2), transparent);
    margin: 4px 0;
  }
  .pipe-content {
    padding-bottom: 2.5rem;
    flex: 1;
  }
  .pipe-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9rem; font-weight: 700;
    color: var(--white); margin-bottom: 0.4rem;
  }
  .pipe-desc { color: var(--slate); font-size: 0.88rem; }

  /* ── File tree ── */
  .tree-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--slate);
    line-height: 2;
    opacity: 0; transform: translateY(20px);
    transition: opacity 0.6s ease, transform 0.6s ease;
  }
  .tree-wrap.reveal { opacity: 1; transform: none; }
  .tree-dir { color: var(--cyan); }
  .tree-py  { color: #A78BFA; }
  .tree-csv { color: #34D399; }
  .tree-npz { color: #FBBF24; }
  .tree-txt { color: var(--slate); }

  /* ── Tech stack ── */
  .stack-grid {
    display: flex; flex-wrap: wrap; gap: 0.75rem;
  }
  .stack-pill {
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--white);
    opacity: 0; transform: scale(0.9);
    transition: opacity 0.4s ease, transform 0.4s ease, box-shadow 0.3s ease;
  }
  .stack-pill.reveal { opacity: 1; transform: none; }
  .stack-pill:hover { box-shadow: var(--glow); border-color: rgba(0,212,255,0.4); }
  .stack-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--cyan); flex-shrink: 0;
  }

  /* ── Code block ── */
  .code-block {
    background: #060910;
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    overflow-x: auto;
    opacity: 0; transform: translateY(20px);
    transition: opacity 0.6s ease, transform 0.6s ease;
  }
  .code-block.reveal { opacity: 1; transform: none; }
  .cb-header {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
  }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
  .dot-r { background: #FF5F56; }
  .dot-y { background: #FFBD2E; }
  .dot-g { background: #27C93F; }
  .cb-fname { color: var(--slate); font-size: 0.75rem; margin-left: auto; }
  .kw { color: #C792EA; }
  .fn { color: #82AAFF; }
  .st { color: #C3E88D; }
  .cm { color: #546E7A; }
  .cy { color: var(--cyan); }

  /* ── Divider ── */
  .divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 0 2rem;
  }

  /* ── Footer ── */
  footer {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--slate);
    font-size: 0.82rem;
    border-top: 1px solid var(--border);
  }
  footer span { color: var(--cyan); }

  /* ── Animations ── */
  @keyframes fadeDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: none; }
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: none; }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
  }
</style>
</head>
<body>

<canvas id="starfield"></canvas>

<!-- ═══════════════════════ HERO ═══════════════════════ -->
<header>
  <div class="badge">⚡ ML · Recommender Systems · Python</div>

  <h1>Amazon Product<br>Recommender System</h1>

  <p class="hero-sub">
    A multi-algorithm recommendation engine built on real Amazon data —
    combining popularity signals, content similarity, collaborative filtering,
    matrix factorisation, and deep learning.
  </p>

  <div class="typewriter">$ python src/content_based.py --product "USB Cable" --top 5</div>

  <div class="pill-row">
    <span class="pill">Python 3.10+</span>
    <span class="pill">scikit-learn</span>
    <span class="pill">Surprise</span>
    <span class="pill">TensorFlow</span>
    <span class="pill">pandas</span>
    <span class="pill">scipy</span>
    <span class="pill">TF-IDF</span>
    <span class="pill">SVD</span>
    <span class="pill">NCF</span>
  </div>

  <div class="neural-wrap">
    <canvas id="neural-canvas" height="220"></canvas>
  </div>
</header>

<!-- ═══════════════════════ ALGORITHMS ═══════════════════════ -->
<section class="section">
  <div class="section-label">Core Modules</div>
  <h2 class="section-title">Six Recommendation Engines</h2>
  <p class="section-desc">Each algorithm tackles the recommendation problem from a different angle — use them standalone or combine them in the hybrid layer.</p>

  <div class="card-grid">
    <div class="algo-card">
      <div class="card-icon">📈</div>
      <div class="card-title">Popularity</div>
      <div class="card-body">Bayesian average scoring that balances mean rating against vote count. Zero cold-start for new users — always has something to suggest.</div>
    </div>
    <div class="algo-card">
      <div class="card-icon">🔍</div>
      <div class="card-title">Content-Based</div>
      <div class="card-body">TF-IDF on product names and descriptions + cosine similarity. Finds textually similar products without needing any user history.</div>
    </div>
    <div class="algo-card">
      <div class="card-icon">👥</div>
      <div class="card-title">Collaborative</div>
      <div class="card-body">User–user and item–item k-NN filtering. Learns from the collective behaviour of similar users to surface hidden gems.</div>
    </div>
    <div class="algo-card">
      <div class="card-icon">🧮</div>
      <div class="card-title">SVD</div>
      <div class="card-body">Matrix factorisation via Singular Value Decomposition. Decomposes the user–item matrix into latent factors for accurate rating prediction.</div>
    </div>
    <div class="algo-card">
      <div class="card-icon">🧠</div>
      <div class="card-title">Neural CF (NCF)</div>
      <div class="card-body">Deep learning model that replaces the dot product with a multi-layer perceptron — learns non-linear user–item interactions.</div>
    </div>
    <div class="algo-card">
      <div class="card-icon">⚡</div>
      <div class="card-title">Hybrid</div>
      <div class="card-body">Weighted ensemble of all models. Adapts scores by user history depth — content-heavy for cold users, CF-heavy for active ones.</div>
    </div>
  </div>
</section>

<hr class="divider"/>

<!-- ═══════════════════════ PIPELINE ═══════════════════════ -->
<section class="section">
  <div class="section-label">Data Pipeline</div>
  <h2 class="section-title">From Raw CSV to Recommendations</h2>
  <p class="section-desc">The preprocessing pipeline handles every messy real-world issue in the Amazon dataset before any model sees the data.</p>

  <div class="pipeline">
    <div class="pipe-step">
      <div class="pipe-left"><div class="pipe-num">01</div><div class="pipe-line"></div></div>
      <div class="pipe-content">
        <div class="pipe-title">Load &amp; Inspect</div>
        <div class="pipe-desc">Read raw <code style="color:var(--cyan)">amazon.csv</code>. All columns ingested as strings first to prevent silent type coercion.</div>
      </div>
    </div>
    <div class="pipe-step">
      <div class="pipe-left"><div class="pipe-num">02</div><div class="pipe-line"></div></div>
      <div class="pipe-content">
        <div class="pipe-title">Clean Numerics</div>
        <div class="pipe-desc">Strip ₹ symbols, commas from prices and rating counts. Handle pipe-separated rating fields. Derive <code style="color:var(--cyan)">discount_amount</code>.</div>
      </div>
    </div>
    <div class="pipe-step">
      <div class="pipe-left"><div class="pipe-num">03</div><div class="pipe-line"></div></div>
      <div class="pipe-content">
        <div class="pipe-title">Explode Multi-User Rows</div>
        <div class="pipe-desc">Each row stores comma-separated user IDs. Exploded to one row per (user, product) interaction — the correct atomic unit for CF.</div>
      </div>
    </div>
    <div class="pipe-step">
      <div class="pipe-left"><div class="pipe-num">04</div><div class="pipe-line"></div></div>
      <div class="pipe-content">
        <div class="pipe-title">Deduplicate &amp; Filter</div>
        <div class="pipe-desc">Keep highest rating per (user, product) pair. Iterative cold-start filter removes users and items with fewer than 2 interactions.</div>
      </div>
    </div>
    <div class="pipe-step">
      <div class="pipe-left"><div class="pipe-num">05</div><div class="pipe-line"></div></div>
      <div class="pipe-content">
        <div class="pipe-title">Encode IDs</div>
        <div class="pipe-desc">Map string user/product IDs to contiguous integers via <code style="color:var(--cyan)">LabelEncoder</code>. Encoders persisted for inference.</div>
      </div>
    </div>
    <div class="pipe-step">
      <div class="pipe-left"><div class="pipe-num">06</div><div class="pipe-line"></div></div>
      <div class="pipe-content">
        <div class="pipe-title">Feature Engineering</div>
        <div class="pipe-desc">TF-IDF (5000 features, bigrams) on product text. MinMax-scaled numeric features. One-hot top-level categories. All saved as sparse artefacts.</div>
      </div>
    </div>
    <div class="pipe-step">
      <div class="pipe-left"><div class="pipe-num">07</div></div>
      <div class="pipe-content" style="padding-bottom:0">
        <div class="pipe-title">Train / Test Split</div>
        <div class="pipe-desc">Leave-one-out split — each user's highest-rated product held out for evaluation. Enables Hit Rate@N metric across all models.</div>
      </div>
    </div>
  </div>
</section>

<hr class="divider"/>

<!-- ═══════════════════════ PROJECT STRUCTURE ═══════════════════════ -->
<section class="section">
  <div class="section-label">Project Layout</div>
  <h2 class="section-title">File Structure</h2>
  <p class="section-desc">Clean separation between data, source modules, saved models, and the serving app.</p>

  <div class="tree-wrap">
    <span class="tree-dir">recommender-system/</span><br>
    &nbsp;├── <span class="tree-dir">data/</span><br>
    &nbsp;│&nbsp;&nbsp; ├── <span class="tree-csv">amazon.csv</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># raw dataset</span><br>
    &nbsp;│&nbsp;&nbsp; └── <span class="tree-dir">processed/</span><br>
    &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── <span class="tree-csv">train.csv</span> <span class="tree-csv">test.csv</span> <span class="tree-csv">products.csv</span><br>
    &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── <span class="tree-npz">interaction_matrix.npz</span>&nbsp;<span class="cm"># sparse user×item</span><br>
    &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├── <span class="tree-npz">tfidf_matrix.npz</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># content features</span><br>
    &nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── <span class="tree-csv">category_dummies.csv</span><br>
    &nbsp;├── <span class="tree-dir">src/</span><br>
    &nbsp;│&nbsp;&nbsp; ├── <span class="tree-py">preprocessing.py</span>&nbsp;&nbsp;&nbsp;<span class="cm"># full ETL pipeline</span><br>
    &nbsp;│&nbsp;&nbsp; ├── <span class="tree-py">popularity.py</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># bayesian scoring</span><br>
    &nbsp;│&nbsp;&nbsp; ├── <span class="tree-py">content_based.py</span>&nbsp;&nbsp;&nbsp;<span class="cm"># TF-IDF + cosine</span><br>
    &nbsp;│&nbsp;&nbsp; ├── <span class="tree-py">collaborative.py</span>&nbsp;&nbsp;&nbsp;<span class="cm"># k-NN CF</span><br>
    &nbsp;│&nbsp;&nbsp; ├── <span class="tree-py">svd.py</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># matrix factorisation</span><br>
    &nbsp;│&nbsp;&nbsp; ├── <span class="tree-py">ncf.py</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># neural CF (MLP)</span><br>
    &nbsp;│&nbsp;&nbsp; ├── <span class="tree-py">hybrid.py</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># weighted ensemble</span><br>
    &nbsp;│&nbsp;&nbsp; └── <span class="tree-py">evaluation.py</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># Hit Rate, NDCG, MAP</span><br>
    &nbsp;├── <span class="tree-dir">models/</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># saved .pkl / .h5</span><br>
    &nbsp;├── <span class="tree-dir">app/</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># Flask / Streamlit UI</span><br>
    &nbsp;├── <span class="tree-dir">notebooks/</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="cm"># EDA &amp; experiments</span><br>
    &nbsp;└── <span class="tree-txt">requirements.txt</span>
  </div>
</section>

<hr class="divider"/>

<!-- ═══════════════════════ QUICK START ═══════════════════════ -->
<section class="section">
  <div class="section-label">Quick Start</div>
  <h2 class="section-title">Get Running in 3 Steps</h2>
  <p class="section-desc">Clone, install, and generate your first recommendations in under a minute.</p>

  <div class="code-block">
    <div class="cb-header">
      <div class="dot dot-r"></div><div class="dot dot-y"></div><div class="dot dot-g"></div>
      <span class="cb-fname">terminal</span>
    </div>
    <span class="cm"># 1. Install dependencies</span><br>
    <span class="kw">pip</span> install -r requirements.txt<br><br>
    <span class="cm"># 2. Preprocess the raw dataset</span><br>
    <span class="kw">python</span> src/<span class="fn">preprocessing.py</span> --data data/amazon.csv<br><br>
    <span class="cm"># 3. Run any recommender</span><br>
    <span class="kw">python</span> src/<span class="fn">popularity.py</span> --top <span class="cy">10</span><br>
    <span class="kw">python</span> src/<span class="fn">content_based.py</span> --product <span class="st">"USB Cable"</span> --top <span class="cy">5</span><br>
    <span class="kw">python</span> src/<span class="fn">collaborative.py</span> --user <span class="st">"AG3D6O4STAQKAY2"</span><br>
    <span class="kw">python</span> src/<span class="fn">hybrid.py</span> --user <span class="st">"AG3D6O4STAQKAY2"</span> --product <span class="st">"USB Cable"</span>
  </div>
</section>

<hr class="divider"/>

<!-- ═══════════════════════ TECH STACK ═══════════════════════ -->
<section class="section">
  <div class="section-label">Built With</div>
  <h2 class="section-title">Tech Stack</h2>
  <p class="section-desc">Production-standard Python ML stack — no unnecessary dependencies.</p>

  <div class="stack-grid">
    <div class="stack-pill"><div class="stack-dot"></div>Python 3.10+</div>
    <div class="stack-pill"><div class="stack-dot"></div>pandas</div>
    <div class="stack-pill"><div class="stack-dot"></div>NumPy</div>
    <div class="stack-pill"><div class="stack-dot"></div>scikit-learn</div>
    <div class="stack-pill"><div class="stack-dot"></div>scipy.sparse</div>
    <div class="stack-pill"><div class="stack-dot"></div>Surprise (SVD)</div>
    <div class="stack-pill"><div class="stack-dot"></div>TensorFlow / Keras</div>
    <div class="stack-pill"><div class="stack-dot"></div>joblib</div>
    <div class="stack-pill"><div class="stack-dot"></div>matplotlib</div>
    <div class="stack-pill"><div class="stack-dot"></div>Flask / Streamlit</div>
  </div>
</section>

<!-- ═══════════════════════ FOOTER ═══════════════════════ -->
<footer>
  Built by <span>Kashish Singhal </span> · Thapar Institute of Engineering &amp; Technology<br>
  Department of Computer Science &amp; Engineering<br><br>
  <span style="color:var(--slate)">Amazon Product Recommender System · 2026</span>
</footer>

<!-- ═══════════════════════ SCRIPTS ═══════════════════════ -->
<script>
// ── Starfield ──────────────────────────────────────────────────────────────
(function(){
  const c = document.getElementById('starfield');
  const ctx = c.getContext('2d');
  let W, H, stars;

  function init(){
    W = c.width  = window.innerWidth;
    H = c.height = window.innerHeight;
    stars = Array.from({length: 160}, () => ({
      x: Math.random()*W, y: Math.random()*H,
      r: Math.random()*1.2+0.2,
      a: Math.random(),
      da: (Math.random()-0.5)*0.005,
      vx: (Math.random()-0.5)*0.12,
      vy: (Math.random()-0.5)*0.06,
    }));
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    stars.forEach(s=>{
      s.x += s.vx; s.y += s.vy;
      s.a  = Math.max(0.05, Math.min(1, s.a+s.da));
      if(s.x<0) s.x=W; if(s.x>W) s.x=0;
      if(s.y<0) s.y=H; if(s.y>H) s.y=0;
      ctx.beginPath();
      ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
      ctx.fillStyle = `rgba(0,212,255,${s.a})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }

  init();
  draw();
  window.addEventListener('resize', init);
})();

// ── Neural recommendation graph ────────────────────────────────────────────
(function(){
  const c   = document.getElementById('neural-canvas');
  const ctx = c.getContext('2d');
  let W, H;

  function resize(){
    W = c.width  = c.offsetWidth;
    H = c.height = 220;
  }
  resize();
  window.addEventListener('resize', resize);

  const PRODUCTS = [
    "USB-C Cable","Wireless Earbuds","Phone Stand",
    "Laptop Sleeve","Screen Wiper","Power Bank",
    "Smart Plug","LED Strip","Webcam","Hub"
  ];

  const nodes = PRODUCTS.map((label, i) => {
    const angle = (i / PRODUCTS.length) * Math.PI * 2 - Math.PI/2;
    const rx = W*0.35, ry = H*0.36;
    return {
      label,
      x: W/2 + rx * Math.cos(angle),
      y: H/2 + ry * Math.sin(angle),
      vx: (Math.random()-0.5)*0.3,
      vy: (Math.random()-0.5)*0.3,
      r: 5,
      pulse: Math.random()*Math.PI*2,
    };
  });

  // Random edges (simulate recommendation connections)
  const edges = [];
  for(let i=0;i<nodes.length;i++){
    const j = (i+1+Math.floor(Math.random()*3)) % nodes.length;
    edges.push({a:i, b:j, progress:0, speed:0.004+Math.random()*0.003});
  }
  // A few cross edges
  edges.push({a:0,b:5,progress:0.5,speed:0.003});
  edges.push({a:2,b:7,progress:0.2,speed:0.004});
  edges.push({a:4,b:9,progress:0.8,speed:0.005});

  let active = 0; // highlighted node index
  let t = 0;

  function lerp(a,b,p){ return a + (b-a)*p; }

  function draw(){
    ctx.clearRect(0,0,W,H);
    t += 0.01;

    // Every ~3s change active node
    if(Math.floor(t) % 3 === 0 && Math.sin(t*Math.PI) > 0.98)
      active = (active + 1) % nodes.length;

    // Draw edges
    edges.forEach(e=>{
      e.progress = (e.progress + e.speed) % 1;
      const A = nodes[e.a], B = nodes[e.b];
      const isActive = e.a===active || e.b===active;

      // Static line
      ctx.beginPath();
      ctx.moveTo(A.x,A.y); ctx.lineTo(B.x,B.y);
      ctx.strokeStyle = isActive
        ? 'rgba(0,212,255,0.25)' : 'rgba(0,212,255,0.06)';
      ctx.lineWidth = isActive ? 1.5 : 0.8;
      ctx.stroke();

      // Travelling dot
      const tx = lerp(A.x,B.x,e.progress);
      const ty = lerp(A.y,B.y,e.progress);
      ctx.beginPath();
      ctx.arc(tx,ty,2.5,0,Math.PI*2);
      ctx.fillStyle = isActive ? '#00D4FF' : 'rgba(0,212,255,0.4)';
      ctx.fill();
    });

    // Draw nodes
    nodes.forEach((n,i)=>{
      n.pulse += 0.04;
      const glow = i===active ? 18 : 8;
      const col  = i===active ? '#00D4FF' : '#2563EB';

      // Outer pulse ring for active
      if(i===active){
        const ring = 8 + Math.sin(n.pulse)*4;
        ctx.beginPath();
        ctx.arc(n.x,n.y,ring,0,Math.PI*2);
        ctx.strokeStyle = 'rgba(0,212,255,0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Node circle
      const grad = ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,glow);
      grad.addColorStop(0, col);
      grad.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
      ctx.fillStyle = col;
      ctx.shadowBlur = glow;
      ctx.shadowColor = col;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label
      ctx.font = i===active ? '600 10px Inter' : '10px Inter';
      ctx.fillStyle = i===active ? '#F0F4FF' : 'rgba(136,146,164,0.8)';
      ctx.textAlign  = n.x < W/2 ? 'right' : 'left';
      ctx.textBaseline = 'middle';
      const pad = i===active ? 12 : 10;
      ctx.fillText(n.label, n.x + (n.x<W/2?-pad:pad), n.y);
    });

    requestAnimationFrame(draw);
  }
  draw();
})();

// ── Scroll reveal ──────────────────────────────────────────────────────────
(function(){
  const observer = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{
      if(e.isIntersecting){
        e.target.classList.add('reveal');
      }
    });
  }, {threshold: 0.15});

  document.querySelectorAll(
    '.section-label,.section-title,.section-desc,' +
    '.algo-card,.pipe-step,.tree-wrap,.code-block,.stack-pill'
  ).forEach(el=>observer.observe(el));

  // Staggered delays for grids
  document.querySelectorAll('.algo-card').forEach((el,i)=>{
    el.style.transition = `opacity 0.5s ease ${i*0.08}s, transform 0.5s ease ${i*0.08}s, box-shadow 0.3s ease, border-color 0.3s ease`;
  });
  document.querySelectorAll('.stack-pill').forEach((el,i)=>{
    el.style.transition = `opacity 0.4s ease ${i*0.05}s, transform 0.4s ease ${i*0.05}s, box-shadow 0.3s ease, border-color 0.3s ease`;
  });
  document.querySelectorAll('.pipe-step').forEach((el,i)=>{
    el.style.transition = `opacity 0.5s ease ${i*0.1}s, transform 0.5s ease ${i*0.1}s`;
  });
})();
</script>
</body>
</html>
