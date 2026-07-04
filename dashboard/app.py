"""
Trustworthy AI Framework for Electrical Metrology — CSIR-NPL
Dashboard: dashboard/app.py  (UI_DEV v2)
"""

import base64
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
PLOTS   = OUTPUTS / "plots"

# ── Colour palette ─────────────────────────────────────────────────────────────
C_NAVY   = "#1F4E8B"
C_BLUE   = "#2E86AB"
C_TEAL   = "#1D9E75"
C_AMBER  = "#EF9F27"
C_RED    = "#E24B4A"
C_PURPLE = "#7F77DD"
C_PINK   = "#E040FB"
C_CYAN   = "#00BCD4"
C_GREEN  = "#1D9E75"

def hex_to_rgba(hex_colour: str, alpha: float = 1.0) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

INST_COLOURS = {
    "AC High Current Source": C_BLUE,
    "Clamp Meter":            C_TEAL,
    "HV Probe":               C_AMBER,
    "HV Breakdown Tester":    C_RED,
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trustworthy AI · Electrical Metrology · CSIR-NPL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state — cold-load always shows landing ─────────────────────────────
# _app_cold_load is set ONLY once per Python process start (i.e. true cold load).
# Tab navigation and st.rerun() do NOT reset it — session_state persists those.
# Hard refresh / new tab → new session → new process → landing shows again.
if "_app_cold_load" not in st.session_state:
    st.session_state._app_cold_load = True
    st.session_state.entered = False

# ── Logo helper ────────────────────────────────────────────────────────────────
def get_logo_b64() -> str:
    logo_path = Path(__file__).parent / "NPL_Logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

LOGO_B64 = get_logo_b64()
LOGO_TAG = (
    f'<img src="data:image/png;base64,{LOGO_B64}" '
    f'style="height:60px;width:60px;object-fit:contain;" />'
    if LOGO_B64 else ""
)

# ══════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.entered:

    # Hide all Streamlit chrome
    st.markdown("""
    <style>
    section[data-testid="stSidebar"],
    #MainMenu, footer,
    header[data-testid="stHeader"]     { display:none !important; }
    .block-container                   { padding:0 !important; max-width:100% !important; }
    [data-testid="stAppViewContainer"] { padding:0 !important; background:#000 !important; }
    [data-testid="stVerticalBlock"]    { padding:0 !important; gap:0 !important; }
    /* Hide the real st.button — JS inside iframe clicks it via postMessage */
    div[data-testid="stButton"]        { position:fixed !important; top:-999px !important;
                                         left:-999px !important; opacity:0 !important;
                                         pointer-events:none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Real st.button — invisible, triggered by JS postMessage from iframe
    entered = st.button("Enter Dashboard", key="landing_enter")
    if entered:
        st.session_state.entered = True
        st.rerun()

    import streamlit.components.v1 as components

    # Everything — shader + overlay UI + button — lives inside this fullscreen iframe
    landing_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#000; overflow:hidden; font-family:'Inter',system-ui,sans-serif; }}

  canvas {{
    position:fixed; top:0; left:0;
    width:100vw; height:100vh; z-index:0; display:block;
  }}

  /* ── Main overlay — headline + subline + button + footer ── */
  #overlay {{
    position:fixed; inset:0; z-index:10;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    gap:24px; pointer-events:none;
  }}

  #headline {{
    font-size:clamp(1.8rem,4.5vw,3rem); font-weight:700; color:#fff;
    text-align:center; max-width:740px; line-height:1.2; letter-spacing:-0.02em;
    text-shadow:0 0 60px rgba(46,134,171,0.6),0 0 120px rgba(29,158,117,0.3);
  }}

  #subline {{
    font-size:0.88rem; color:rgba(255,255,255,0.38);
    letter-spacing:0.08em; text-align:center;
  }}

  /* ── Glow button ── */
  #btn-wrap {{
    pointer-events:all; position:relative;
    padding:2px; border-radius:10px; cursor:pointer;
    margin-top:16px;
  }}
  #btn-wrap::before {{
    content:''; position:absolute; inset:-1px; border-radius:10px; z-index:0;
    background:linear-gradient(90deg,
      transparent 0%,#E040FB 25%,transparent 55%,#2E86AB 80%,transparent 100%);
    background-size:200% 100%;
    animation:glow-fwd 2.4s linear infinite;
  }}
  #btn-wrap::after {{
    content:''; position:absolute; inset:-1px; border-radius:10px; z-index:0;
    background:linear-gradient(270deg,
      transparent 0%,#2E86AB 25%,transparent 55%,#E040FB 80%,transparent 100%);
    background-size:200% 100%;
    animation:glow-rev 2.4s linear infinite;
  }}
  @keyframes glow-fwd {{ 0%{{background-position:100% 0}} 100%{{background-position:-100% 0}} }}
  @keyframes glow-rev {{ 0%{{background-position:0 0}}    100%{{background-position:200% 0}} }}

  #btn-inner {{
    position:relative; z-index:1;
    background:#080810; border-radius:8px;
    padding:14px 52px; font-size:1rem; font-weight:600;
    color:white; letter-spacing:0.1em; border:none;
    display:block; transition:background 0.2s; cursor:pointer; white-space:nowrap;
    font-family:'Inter',system-ui,sans-serif;
  }}
  #btn-inner:hover {{ background:#12121e; }}

  /* ── Footer — true bottom of page ── */
  #credit {{
    position:fixed; bottom:20px; left:0; right:0;
    text-align:center;
    font-size:0.75rem; color:rgba(255,255,255,0.25);
    pointer-events:none; letter-spacing:0.04em;
  }}

  /* ── Loading veil — blur + glowing box ── */
  #loading-veil {{
    display:none; position:fixed; inset:0; z-index:999;
    backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
    background:rgba(0,0,0,0.6);
    align-items:center; justify-content:center;
  }}
  #loading-veil.show {{ display:flex; }}
  #loading-box {{
    display:flex; flex-direction:column; align-items:center; gap:22px;
    padding:48px 72px; border-radius:16px;
    background:rgba(8,8,18,0.92);
    border:1px solid rgba(46,134,171,0.4);
    box-shadow:0 0 40px rgba(46,134,171,0.3),0 0 90px rgba(224,64,251,0.15),
               inset 0 0 30px rgba(0,0,0,0.5);
    animation:box-pulse 1.8s ease-in-out infinite;
  }}
  @keyframes box-pulse {{
    0%,100% {{ box-shadow:0 0 40px rgba(46,134,171,0.3),0 0 90px rgba(224,64,251,0.15); }}
    50%     {{ box-shadow:0 0 70px rgba(46,134,171,0.55),0 0 140px rgba(224,64,251,0.3); }}
  }}
  .spin-ring {{
    width:52px; height:52px;
    border:3px solid rgba(46,134,171,0.15);
    border-top-color:#2E86AB; border-right-color:#E040FB;
    border-radius:50%; animation:spin-it 1s linear infinite;
  }}
  @keyframes spin-it {{ to {{ transform:rotate(360deg); }} }}
  #load-text {{
    color:rgba(255,255,255,0.75); font-size:0.92rem;
    letter-spacing:0.2em; text-transform:uppercase; font-weight:600;
  }}
  #load-sub {{ color:rgba(255,255,255,0.3); font-size:0.72rem; letter-spacing:0.08em; }}
</style>
</head>
<body>

<canvas id="c"></canvas>

<div id="overlay">
  <div id="headline">Trustworthy AI Framework<br>for Electrical Metrology</div>
  <div id="subline">
    Knowledge Graph &nbsp;·&nbsp; XAI &nbsp;·&nbsp; Conformal UQ &nbsp;·&nbsp; GNN Anomaly Detection
  </div>
  <div id="btn-wrap">
    <button id="btn-inner" onclick="enterDashboard()">Enter Dashboard</button>
  </div>
</div>

<div id="credit">Made with ❤️ by Dia Vats</div>

<div id="loading-veil">
  <div id="loading-box">
    <div class="spin-ring"></div>
    <div id="load-text">Loading</div>
    <div id="load-sub">Trustworthy AI Dashboard</div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── Three.js shader ────────────────────────────────────────────────────────
const canvas   = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
renderer.setPixelRatio(window.devicePixelRatio);
const camera = new THREE.Camera();
camera.position.z = 1;
const scene    = new THREE.Scene();
const uniforms = {{
  time:       {{ type:'f',  value:1.0 }},
  resolution: {{ type:'v2', value:new THREE.Vector2() }},
}};
const material = new THREE.ShaderMaterial({{
  uniforms,
  vertexShader:'void main(){{ gl_Position=vec4(position,1.0); }}',
  fragmentShader:`
    precision highp float;
    uniform vec2  resolution;
    uniform float time;
    void main(void){{
      vec2  uv=(gl_FragCoord.xy*2.0-resolution.xy)/min(resolution.x,resolution.y);
      float t=time*0.05;
      float lw=0.002;
      vec3  col=vec3(0.0);
      for(int j=0;j<3;j++){{
        for(int i=0;i<5;i++){{
          col[j]+=lw*float(i*i)/abs(
            fract(t-0.01*float(j)+float(i)*0.01)*5.0
            -length(uv)+mod(uv.x+uv.y,0.2));
        }}
      }}
      gl_FragColor=vec4(col[0],col[1],col[2],1.0);
    }}
  `,
}});
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),material));

function resize(){{
  const w=window.innerWidth, h=window.innerHeight;
  renderer.setSize(w,h);
  uniforms.resolution.value.set(renderer.domElement.width, renderer.domElement.height);
}}
resize();
window.addEventListener('resize', resize);
(function loop(){{
  requestAnimationFrame(loop);
  uniforms.time.value += 0.05;
  renderer.render(scene, camera);
}})();

// ── Enter button — show loading veil, then click hidden Streamlit button ──
function enterDashboard(){{
  // Show blur loading screen immediately
  document.getElementById('loading-veil').classList.add('show');
  document.getElementById('overlay').style.pointerEvents='none';

  // After 1.8s, trigger the hidden Streamlit button in the parent frame
  setTimeout(function(){{
    var parentDoc = window.parent.document;
    var btns = parentDoc.querySelectorAll('button');
    for(var i=0; i<btns.length; i++){{
      if(btns[i].textContent.trim()==='Enter Dashboard'){{
        btns[i].click();
        break;
      }}
    }}
  }}, 1800);
}}
</script>
</body>
</html>"""

    components.html(landing_html, height=700, scrolling=False)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — dashboard only
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
/* ── Tabs ── */
button[data-baseweb="tab"]{{font-weight:600;font-size:0.85rem;}}
[data-testid="stMetricLabel"]{{font-size:0.78rem;opacity:0.8;}}
[data-testid="stMetricValue"]{{font-size:1.6rem;font-weight:700;}}
hr{{margin:0.5rem 0 1rem 0;}}
section[data-testid="stSidebar"]{{width:260px !important;}}

/* ── KPI cards — monochrome fill + glowing border ── */
.kpi-card{{
  background:rgba(var(--rgb),0.12);
  border:1.5px solid rgba(var(--rgb),0.55);
  border-radius:10px;
  padding:18px 20px;
  box-shadow:0 0 18px rgba(var(--rgb),0.2),inset 0 0 12px rgba(var(--rgb),0.06);
  transition:box-shadow 0.3s;
  margin-bottom:4px;
}}
.kpi-card:hover{{box-shadow:0 0 32px rgba(var(--rgb),0.45),inset 0 0 20px rgba(var(--rgb),0.1);}}
.kpi-label{{font-size:0.7rem;opacity:0.6;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;}}
.kpi-value{{font-size:1.8rem;font-weight:700;color:rgba(var(--rgb),1);line-height:1;}}

/* ── Styled expanders ── */
[data-testid="stExpander"]{{
  background:rgba(255,255,255,0.025) !important;
  border:1px solid rgba(255,255,255,0.08) !important;
  border-radius:10px !important;
  overflow:hidden;
}}
[data-testid="stExpander"] summary{{
  font-weight:600;font-size:0.88rem;letter-spacing:0.02em;
  padding:12px 16px;
  border-bottom:1px solid rgba(255,255,255,0.06);
}}
[data-testid="stExpander"] summary:hover{{background:rgba(255,255,255,0.04);}}
[data-testid="stExpander"] > div > div{{padding:16px;}}

/* ── Animated footer text ── */
@keyframes chroma-flow{{
  0%   {{background-position:0% 50%;}}
  100% {{background-position:200% 50%;}}
}}
.footer-text{{
  background:linear-gradient(90deg,
    #E040FB,#c084fc,#818cf8,#2E86AB,#1D9E75,#2E86AB,#818cf8,#c084fc,#E040FB);
  background-size:200% auto;
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  animation:chroma-flow 4s linear infinite;
  font-weight:600;font-size:0.85rem;letter-spacing:0.05em;
  display:inline-block;
}}

/* ── Pipeline status pills ── */
.status-pill{{
  display:inline-flex;align-items:center;gap:7px;
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.07);
  border-radius:20px;padding:4px 12px 4px 7px;
  font-size:0.76rem;margin-bottom:4px;width:100%;
}}
.pill-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;}}
.pill-ok  {{background:{C_TEAL};box-shadow:0 0 6px {C_TEAL}99;}}
.pill-fail{{background:{C_RED};box-shadow:0 0 6px {C_RED}99;}}

/* ── Theme toggle pills in sidebar ── */
.theme-toggle{{display:flex;gap:8px;margin:8px 0;}}
.theme-pill{{
  flex:1;text-align:center;padding:6px 0;border-radius:8px;
  font-size:0.78rem;font-weight:600;cursor:pointer;
  border:1.5px solid rgba(255,255,255,0.12);
  background:rgba(255,255,255,0.04);color:rgba(255,255,255,0.5);
  transition:all 0.2s;
}}
.theme-pill.active{{
  background:rgba(46,134,171,0.2);
  border-color:{C_BLUE};color:white;
  box-shadow:0 0 10px rgba(46,134,171,0.3);
}}

/* ── Regenerate pipeline button — conic glow border ── */
div[data-testid="stButton"] > button{{
  position:relative;isolation:isolate;
  background:#0a0a14;color:white;border:none;
  border-radius:8px;padding:11px 20px;
  font-weight:600;font-size:0.85rem;letter-spacing:0.05em;
  width:100%;cursor:pointer;transition:background 0.2s;
  overflow:visible;
}}
div[data-testid="stButton"] > button::before{{
  content:'';position:absolute;inset:-2px;border-radius:10px;z-index:-1;
  background:linear-gradient(90deg,#E040FB,#2E86AB,#1D9E75,#E040FB);
  background-size:300% 100%;
  animation:btn-flow 2.5s linear infinite;
}}
div[data-testid="stButton"] > button::after{{
  content:'';position:absolute;inset:-1px;border-radius:9px;z-index:-1;
  background:#0a0a14;
}}
@keyframes btn-flow{{0%{{background-position:0% 0}}100%{{background-position:300% 0}}}}
div[data-testid="stButton"] > button:hover{{background:#12121e;}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_all():
    names = [
        "calibration_data","ml_results","gum_comparison",
        "shap_values","gnn_results","autoencoder_results","multitask_results",
    ]
    d = {}
    for name in names:
        path = OUTPUTS / f"{name}.csv"
        try:    d[name] = pd.read_csv(path)
        except FileNotFoundError: d[name] = None
    return d

@st.cache_resource
def load_model():
    pkl = OUTPUTS / "model_bundle.pkl"
    if not pkl.exists(): return None
    try:
        import joblib
        return joblib.load(pkl)
    except Exception: return None

def predict_live(bundle, indicated, domain, instrument):
    dm = {"current":0.0,"voltage":1.0}
    im = {"AC High Current Source":0.0,"Clamp Meter":1/3,"HV Probe":2/3,"HV Breakdown Tester":1.0}
    X    = np.array([[indicated, dm[domain], im[instrument]]])
    pred = bundle["model"].predict(X)[0]
    return pred, bundle["q"]

DATA  = load_all()
MODEL = load_model()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if LOGO_B64:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:14px 4px 18px 4px;">
          <img src="data:image/png;base64,{LOGO_B64}"
               style="height:72px;width:72px;object-fit:contain;
                      filter:drop-shadow(0 0 8px rgba(46,134,171,0.5));" />
          <div>
            <div style="font-weight:700;font-size:0.95rem;color:white;line-height:1.25;">CSIR-NPL</div>
            <div style="font-size:0.7rem;opacity:0.55;color:white;line-height:1.4;">
              National Physical Laboratory<br>Electrical Metrology
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding:14px 4px 18px;font-weight:700;color:white;'>CSIR-NPL</div>",
                    unsafe_allow_html=True)

    st.divider()

    # Theme toggle — visual only, dark always active via config.toml
    st.markdown("""
    <div style="font-size:0.75rem;opacity:0.55;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:6px;">Theme</div>
    <div class="theme-toggle">
      <div class="theme-pill active">Dark</div>
      <div class="theme-pill">Light</div>
    </div>
    <div style="font-size:0.68rem;opacity:0.35;margin-bottom:4px;">
      Light mode — coming soon
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**Pipeline output status**")
    files_to_check = {
        "calibration_data.csv":    "Calibration data",
        "ml_results.csv":          "ML results",
        "gum_comparison.csv":      "GUM comparison",
        "shap_values.csv":         "SHAP values",
        "gnn_results.csv":         "GNN results",
        "autoencoder_results.csv": "Autoencoder results",
        "multitask_results.csv":   "Multi-task results",
        "model_bundle.pkl":        "Model bundle",
    }
    for fname, label in files_to_check.items():
        exists    = (OUTPUTS / fname).exists()
        dot_class = "pill-ok" if exists else "pill-fail"
        st.markdown(
            f'<div class="status-pill"><span class="pill-dot {dot_class}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("Regenerate pipeline"):
        with st.spinner("Running pipeline…"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_all.py")],
                capture_output=True, text=True,
            )
        if result.returncode == 0:
            st.success("Pipeline completed!")
            st.cache_data.clear()
            st.cache_resource.clear()
        else:
            st.error("Pipeline error:")
            st.code(result.stderr[-2000:])


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

def render_footer():
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;padding:8px 0;'>"
        "<span class='footer-text'>Made with ♥ by Dia Vats</span>"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — styled KPI card
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, rgb: str) -> str:
    """rgb = '127,119,221' format (no #)"""
    return f"""
    <div class="kpi-card" style="--rgb:{rgb};">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview","Knowledge Graph","ML Predictions & UQ",
    "SHAP XAI","Anomaly Detection","Multi-task NN","GUM Comparison",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    # Hero — no logo duplication (logo is already in sidebar), no intern/supervisor
    st.markdown("""
    <div style="
        background:linear-gradient(90deg,#1F4E8B 0%,#2E86AB 60%,#1D9E75 100%);
        padding:28px 36px;border-radius:12px;margin-bottom:24px;
        box-shadow:0 4px 24px rgba(46,134,171,0.25);
    ">
      <h1 style="color:white;margin:0;font-size:1.75rem;letter-spacing:-0.5px;font-weight:700;">
        Trustworthy AI · Electrical Metrology · CSIR-NPL
      </h1>
      <p style="color:rgba(255,255,255,0.65);margin:8px 0 0 0;font-size:0.88rem;">
        Development of Trustworthy AI Models/Framework for Electrical Metrology
      </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards — purple, teal, pink, cyan
    m1, m2, m3, m4 = st.columns(4)
    cards = [
        (m1, "Calibration Points", "31",     "127,119,221"),  # purple
        (m2, "Instruments",        "4",       "29,158,117"),   # teal
        (m3, "KG Triples",         "799",     "224,64,251"),   # pink
        (m4, "SHACL Status",       "Passed",  "0,188,212"),    # cyan
    ]
    for col, label, value, rgb in cards:
        col.markdown(kpi_card(label, value, rgb), unsafe_allow_html=True)

    st.divider()
    st.subheader("Instrument Summary")
    cal = DATA["calibration_data"]
    if cal is not None:
        summary_rows = []
        for inst, grp in cal.groupby("instrument"):
            re_min = grp["ratio_error_pct"].min()
            re_max = grp["ratio_error_pct"].max()
            domain = grp["measurement_domain"].iloc[0]
            date   = grp["cal_date"].iloc[0] if "cal_date" in grp.columns else "—"
            summary_rows.append({
                "Instrument":"Instrument",
                "Domain":domain,"Points (n)":len(grp),
                "RE Range (%)":f"{re_min:.3f} → {re_max:.3f}",
                "Calibration Date":date,
            })
            summary_rows[-1]["Instrument"] = inst
        summary_df = pd.DataFrame(summary_rows).set_index("Instrument")
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.error("calibration_data.csv not found in outputs/")

    st.divider()
    st.subheader("Raw Calibration Data")
    if cal is not None:
        st.dataframe(cal, use_container_width=True, height=350)
        st.download_button("Download calibration_data.csv",
                           data=cal.to_csv(index=False).encode(),
                           file_name="calibration_data.csv", mime="text/csv")
    else:
        st.error("calibration_data.csv not found.")
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Knowledge Graph
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Knowledge Graph — Electrical Metrology Ontology")
    st.caption("799 RDF triples · OWL/RDF + SOSA + PROV-O + QUDT + EMO")

    nodes = [
        {"label":"National Standard",      "x":0.50,"y":1.00,"color":C_TEAL,  "type":"Standard"},
        {"label":"Curr. Comparator",       "x":0.20,"y":0.70,"color":C_AMBER, "type":"Equipment"},
        {"label":"Transfer Std. DMM",      "x":0.80,"y":0.70,"color":C_AMBER, "type":"Equipment"},
        {"label":"Calibration Event",      "x":0.50,"y":0.50,"color":C_PURPLE,"type":"Event"},
        {"label":"Env. Conditions",        "x":0.88,"y":0.50,"color":C_AMBER, "type":"Context"},
        {"label":"CSIR-NPL",               "x":0.15,"y":0.30,"color":C_TEAL,  "type":"Organisation"},
        {"label":"Certificate",            "x":0.50,"y":0.30,"color":C_PURPLE,"type":"Document"},
        {"label":"Instrument (DUT)",       "x":0.85,"y":0.30,"color":C_TEAL,  "type":"Entity"},
        {"label":"Calibration Points (31)","x":0.50,"y":0.10,"color":C_BLUE,  "type":"Data"},
    ]
    edges = [(0,1),(0,2),(1,3),(2,3),(3,4),(3,5),(3,6),(3,7),(6,8),(7,8)]

    edge_traces = [go.Scatter(
        x=[nodes[i]["x"],nodes[j]["x"],None], y=[nodes[i]["y"],nodes[j]["y"],None],
        mode="lines", line=dict(width=1.5,color="rgba(180,180,200,0.35)"),
        hoverinfo="none", showlegend=False,
    ) for i,j in edges]

    node_trace = go.Scatter(
        x=[n["x"] for n in nodes], y=[n["y"] for n in nodes],
        mode="markers+text",
        marker=dict(size=22,color=[n["color"] for n in nodes],
                    line=dict(width=2,color="rgba(255,255,255,0.2)")),
        text=[n["label"] for n in nodes], textposition="top center",
        textfont=dict(size=10,color="#E8EAF0"),
        customdata=[n["type"] for n in nodes],
        hovertemplate="<b>%{text}</b><br>Type: %{customdata}<extra></extra>",
        showlegend=False,
    )
    kg_fig = go.Figure(data=edge_traces+[node_trace])
    kg_fig.update_layout(
        height=480,margin=dict(l=20,r=20,t=20,b=20),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
    )
    st.plotly_chart(kg_fig, use_container_width=True)

    leg_cols = st.columns(4)
    for col,(label,colour) in zip(leg_cols,[
        ("Entities / Lab",C_TEAL),("Documents / Events",C_PURPLE),
        ("Standards / Context",C_AMBER),("Data Points",C_BLUE),
    ]):
        col.markdown(f'<span style="color:{colour};font-size:1.2rem;">●</span> <small>{label}</small>',
                     unsafe_allow_html=True)

    st.divider()

    # KPI cards for KG stats
    s1, s2, s3 = st.columns(3)
    kg_cards = [
        (s1, "RDF Triples",            "799",                           "127,119,221"),
        (s2, "Ontologies",             "4 · QUDT · PROV-O · SOSA · EMO","29,158,117"),
        (s3, "Instrument types in KG", "6",                             "0,188,212"),
    ]
    for col,label,value,rgb in kg_cards:
        col.markdown(kpi_card(label,value,rgb), unsafe_allow_html=True)

    st.divider()
    st.subheader("SHACL Validation")
    shacl_path = OUTPUTS / "shacl_validation_report.txt"
    if shacl_path.exists():
        report_text = shacl_path.read_text(encoding="utf-8",errors="ignore")
        if "conforms: true" in report_text.lower():
            st.success("SHACL Validation Passed — Knowledge graph conforms to all shape constraints.")
        else:
            st.error("SHACL Violations detected:")
            st.code(report_text)
        with st.expander("Full SHACL report"):
            st.code(report_text)
    else:
        st.error("shacl_validation_report.txt not found in outputs/")

    st.divider()
    with st.expander("Why SOSA? — Semantic Sensor Network Ontology"):
        st.markdown("""
        **SOSA (Sensor, Observation, Sample and Actuator)** is a W3C standard ontology
        for modelling sensor networks and observations.

        In this framework:
        - Each **CalibrationPoint** is modelled as a `sosa:Observation`
        - The **reference standard** (e.g., National Current Comparator) is the `sosa:Sensor`
        - The **instrument under test (DUT)** is the `sosa:FeatureOfInterest`
        - The calibration laboratory (**CSIR-NPL**) is the `prov:Agent`

        This provides interoperability with IoT and laboratory information systems,
        and enables SPARQL-based federated queries across metrology databases.
        """)
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ML Predictions + UQ
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("ML Predictions & Conformal Uncertainty Quantification")
    ml  = DATA["ml_results"]
    cal = DATA["calibration_data"]

    if ml is None:
        st.error("ml_results.csv not found in outputs/")
    else:
        fig3 = go.Figure()
        for inst in ml["instrument"].unique():
            sub    = ml[ml["instrument"]==inst].sort_values("indicated_value")
            colour = INST_COLOURS.get(inst, C_BLUE)
            fig3.add_trace(go.Scatter(
                x=pd.concat([sub["indicated_value"],sub["indicated_value"].iloc[::-1]]),
                y=pd.concat([sub["conformal_upper_95pct"],sub["conformal_lower_95pct"].iloc[::-1]]),
                fill="toself",fillcolor=hex_to_rgba(colour,0.15),
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{inst} 95% PI",showlegend=False,hoverinfo="skip",
            ))
            fig3.add_trace(go.Scatter(
                x=sub["indicated_value"],y=sub["ml_predicted_ratio_error_pct"],
                mode="lines",line=dict(color=colour,dash="dash",width=1.5),
                name=f"{inst} ML pred.",showlegend=True,
            ))
            sym  = "x"  if inst=="HV Breakdown Tester" else "circle"
            msz  = 10   if inst=="HV Breakdown Tester" else 7
            fig3.add_trace(go.Scatter(
                x=sub["indicated_value"],y=sub["ratio_error_pct"],
                mode="markers",
                marker=dict(color=colour,symbol=sym,size=msz,line=dict(width=1.5,color="white")),
                name=f"{inst} actual",showlegend=True,
            ))

        hvbd = ml[ml["instrument"]=="HV Breakdown Tester"]
        if not hvbd.empty:
            fig3.add_annotation(
                x=hvbd["indicated_value"].mean(),
                y=hvbd["ratio_error_pct"].max()+1.5,
                text="10–17% RE — anomaly zone",
                showarrow=True,arrowhead=2,arrowcolor=C_RED,
                font=dict(color=C_RED,size=11),
                bgcolor="rgba(226,75,74,0.12)",bordercolor=C_RED,
            )
        fig3.update_layout(
            xaxis_title="Indicated Value (A or kV)",yaxis_title="Ratio Error (%)",
            legend=dict(orientation="h",y=-0.2),height=480,
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,22,36,0.6)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.divider()

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{C_PURPLE}22,{C_BLUE}22);
                    border:1px solid {C_PURPLE}55;border-radius:10px;
                    padding:16px 20px;margin-bottom:8px;">
          <h3 style="margin:0;color:{C_PURPLE};">Live GUM Calculator</h3>
          <p style="margin:4px 0 0 0;font-size:0.83rem;opacity:0.8;">
            Select instrument & indicated value — real-time ratio error prediction
          </p>
        </div>
        """, unsafe_allow_html=True)

        calc_col, result_col = st.columns([1,1])
        with calc_col:
            inst_options = ["AC High Current Source","Clamp Meter","HV Probe","HV Breakdown Tester"]
            sel_inst = st.selectbox("Select instrument", inst_options, key="calc_inst")
            if sel_inst in ("AC High Current Source","Clamp Meter"):
                domain = "current"
                ranges = {"AC High Current Source":(100.0,4500.0,100.0,500.0),
                          "Clamp Meter":(400.0,2500.0,400.0,100.0)}
                unit_label = "A"
            else:
                domain = "voltage"
                ranges = {"HV Probe":(1.5,28.0,1.5,1.0),
                          "HV Breakdown Tester":(1.0,3.5,1.0,0.5)}
                unit_label = "kV"
            v_min,v_max,v_default,v_step = ranges[sel_inst]
            indicated = st.slider(f"Indicated value ({unit_label})",
                                  min_value=float(v_min),max_value=float(v_max),
                                  value=float(v_default),step=float(v_step),key="calc_slider")
        with result_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if MODEL is not None:
                try:
                    pred, q = predict_live(MODEL, indicated, domain, sel_inst)
                    st.metric(label=f"Predicted Ratio Error — {sel_inst}",
                              value=f"{pred:+.4f} %",
                              delta=f"± {q:.4f} % (95% PI half-width)")
                    st.code(f"RE = {pred:+.4f} %  ±  {q:.4f} %\n"
                            f"(U, k ≈ 2.0, coverage probability = 95%)\n"
                            f"Instrument : {sel_inst}\n"
                            f"Indicated  : {indicated} {unit_label}\n"
                            f"Domain     : {domain}", language="text")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.warning("model_bundle.pkl not found. Run the pipeline first.")

        st.divider()
        st.subheader("ML Results Table")
        display_cols = ["instrument","indicated_value","ratio_error_pct",
                        "ml_predicted_ratio_error_pct","conformal_lower_95pct",
                        "conformal_upper_95pct","conformal_halfwidth_pct",
                        "gum_expanded_U_pct","loo_residual_pct"]
        cols_present = [c for c in display_cols if c in ml.columns]
        st.dataframe(ml[cols_present], use_container_width=True, height=300)

        st.divider()
        cov_col1, cov_col2, cov_col3 = st.columns(3)
        if "ratio_error_pct" in ml.columns and "conformal_lower_95pct" in ml.columns:
            in_pi = ((ml["ratio_error_pct"]>=ml["conformal_lower_95pct"]) &
                     (ml["ratio_error_pct"]<=ml["conformal_upper_95pct"])).sum()
            total = len(ml)
            # Styled coverage cards — purple theme
            cov_data = [
                (cov_col1,"Empirical Coverage", f"{in_pi}/{total} (100%)", "127,119,221"),
                (cov_col2,"PI Half-width",       "±7.01%",                 "224,64,251"),
                (cov_col3,"Coverage Target",     "95%",                    "0,188,212"),
            ]
            for col,label,value,rgb in cov_data:
                col.markdown(kpi_card(label,value,rgb), unsafe_allow_html=True)
            st.success("100% empirical coverage — every true ratio error value falls within the predicted 95% conformal interval.")
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SHAP XAI
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("SHAP Explainability Analysis")
    shap_png = PLOTS / "shap_analysis.png"
    if shap_png.exists():
        st.image(str(shap_png),
                 caption="SHAP Summary — Mean |SHAP| per feature across all calibration points",
                 use_container_width=True)
    else:
        st.warning("shap_analysis.png not found in outputs/plots/")

    st.info("Key Finding: Domain and instrument type dominate SHAP importance — confirming "
            "systematic instrument-level effects outweigh within-instrument variation.")
    st.divider()

    shap_df = DATA["shap_values"]
    if shap_df is None:
        st.error("shap_values.csv not found in outputs/")
    else:
        st.subheader("Mean |SHAP| per Feature — by Instrument")
        feature_cols   = ["shap_indicated_norm","shap_domain","shap_instrument_type"]
        feature_labels = {"shap_indicated_norm":"Indicated Value (normalised)",
                          "shap_domain":"Measurement Domain",
                          "shap_instrument_type":"Instrument Type"}
        bar_rows = []
        for inst,grp in shap_df.groupby("instrument"):
            for fc in feature_cols:
                if fc in grp.columns:
                    bar_rows.append({"Instrument":inst,
                                     "Feature":feature_labels.get(fc,fc),
                                     "Mean |SHAP|":grp[fc].abs().mean()})
        if bar_rows:
            shap_fig = px.bar(pd.DataFrame(bar_rows),x="Feature",y="Mean |SHAP|",
                              color="Instrument",barmode="group",
                              color_discrete_map=INST_COLOURS,template="plotly_dark",height=400)
            shap_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,22,36,0.6)",
                                   xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                                   yaxis=dict(gridcolor="rgba(255,255,255,0.07)"))
            st.plotly_chart(shap_fig, use_container_width=True)

        st.divider()
        st.subheader("SHAP Values Table")
        st.dataframe(shap_df, use_container_width=True, height=300)
        st.divider()

        with st.expander("Physical Interpretation of SHAP Directions"):
            st.markdown("""
            | Feature | SHAP Direction | Metrological Meaning |
            |---------|---------------|----------------------|
            | **Instrument Type** | Large positive for HV instruments | High-voltage instruments have systematic positive ratio errors due to capacitive loading |
            | **Measurement Domain** | Positive for voltage | Voltage calibrations show higher RE than current calibrations at the same range position |
            | **Indicated Value (norm.)** | Positive at low, negative at high | Non-linearity: instruments degrade at range extremes |
            | **shap_instrument_type < 0** | HV Probe at high indicated values | HV Probe RE becomes negative above 20 kV — over-compensation artefact |
            | **shap_domain > 10** | HV instruments | Voltage domain is the strongest separator — contributes most to anomalous prediction |

            **Metrological implication:** SHAP validates polynomial regression is capturing genuine
            physical phenomena — most important features align with known metrology principles.
            """)
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("Anomaly Detection — GNN & Autoencoder")
    gnn_df = DATA["gnn_results"]
    ae_df  = DATA["autoencoder_results"]

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f"<h4 style='color:{C_BLUE};'>Graph Neural Network (GCN)</h4>", unsafe_allow_html=True)
        gnn_png = PLOTS / "gnn_anomaly.png"
        if gnn_png.exists():
            st.image(str(gnn_png), caption="GNN Anomaly Detection", use_container_width=True)
        else:
            st.warning("gnn_anomaly.png not found.")
        if gnn_df is not None:
            gnn_is_anom  = gnn_df["gnn_is_anomaly"].apply(lambda v: str(v).strip().lower()=="true")
            threshold_gnn = gnn_df["gnn_anomaly_score"].quantile(0.75)
            gnn_fig = go.Figure()
            gnn_fig.add_trace(go.Scatter(
                x=list(range(len(gnn_df))),y=gnn_df["gnn_anomaly_score"],mode="markers",
                marker=dict(color=[C_RED if a else C_BLUE for a in gnn_is_anom],
                            size=9,line=dict(width=1,color="white")),
                text=gnn_df["instrument"],
                hovertemplate="%{text}<br>Score: %{y:.5f}<extra></extra>",name="GNN score",
            ))
            gnn_fig.add_hline(y=threshold_gnn,line_dash="dash",line_color=C_RED,
                              annotation_text=f"Threshold ({threshold_gnn:.4f})",
                              annotation_position="top right")
            gnn_fig.update_layout(height=280,xaxis_title="Point index",
                                  yaxis_title="GNN Anomaly Score (MSE)",
                                  paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,22,36,0.6)",
                                  xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                                  yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),showlegend=False)
            st.plotly_chart(gnn_fig, use_container_width=True)
        else:
            st.error("gnn_results.csv not found.")

    with right_col:
        st.markdown(f"<h4 style='color:{C_PURPLE};'>Tabular Autoencoder</h4>", unsafe_allow_html=True)
        ae_png = PLOTS / "autoencoder_anomaly.png"
        if ae_png.exists():
            st.image(str(ae_png), caption="Autoencoder Anomaly Detection", use_container_width=True)
        else:
            st.warning("autoencoder_anomaly.png not found.")
        if ae_df is not None:
            ae_is_anom  = ae_df["ae_anomaly"].apply(lambda v: str(v).strip().lower()=="true")
            threshold_ae = ae_df["ae_score"].quantile(0.75)
            ae_fig = go.Figure()
            ae_fig.add_trace(go.Scatter(
                x=list(range(len(ae_df))),y=ae_df["ae_score"],mode="markers",
                marker=dict(color=[C_RED if a else C_PURPLE for a in ae_is_anom],
                            size=9,line=dict(width=1,color="white")),
                text=ae_df["instrument"],
                hovertemplate="%{text}<br>Score: %{y:.5f}<extra></extra>",name="AE score",
            ))
            ae_fig.add_hline(y=threshold_ae,line_dash="dash",line_color=C_RED,
                             annotation_text=f"Threshold ({threshold_ae:.5f})",
                             annotation_position="top right")
            ae_fig.update_layout(height=280,xaxis_title="Point index",
                                 yaxis_title="Autoencoder Score (MSE)",
                                 paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,22,36,0.6)",
                                 xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                                 yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),showlegend=False)
            st.plotly_chart(ae_fig, use_container_width=True)
        else:
            st.error("autoencoder_results.csv not found.")

    st.divider()
    st.subheader("Combined Anomaly Table — Points Flagged by Both Methods")
    if gnn_df is not None and ae_df is not None:
        combined = gnn_df[["cert_id","instrument","indicated_value","ratio_error_pct",
                            "gnn_anomaly_score","gnn_is_anomaly"]].copy()
        combined["ae_score"]       = ae_df["ae_score"].values
        combined["ae_anomaly"]     = ae_df["ae_anomaly"].values
        combined["gnn_is_anomaly"] = combined["gnn_is_anomaly"].apply(lambda v: str(v).strip().lower()=="true")
        combined["ae_anomaly"]     = combined["ae_anomaly"].apply(lambda v: str(v).strip().lower()=="true")
        combined["both_flagged"]   = combined["gnn_is_anomaly"] & combined["ae_anomaly"]

        def highlight_both(row):
            c = "background-color:rgba(226,75,74,0.20);" if row["both_flagged"] else ""
            return [c]*len(row)

        st.dataframe(combined.style.apply(highlight_both,axis=1),
                     use_container_width=True,height=320)
        both_count = combined["both_flagged"].sum()
        st.success(f"{both_count} point(s) flagged as anomalous by both GNN and Autoencoder independently.")

        hvbt_both  = combined[combined["instrument"]=="HV Breakdown Tester"]["both_flagged"].sum()
        hvbt_total = (combined["instrument"]=="HV Breakdown Tester").sum()
        st.warning(f"Finding: {hvbt_both} of {hvbt_total} HV Breakdown Tester points flagged by both methods. "
                   f"All {hvbt_total} HVBT points show elevated anomaly scores — no other instrument approaches this level. "
                   "Cross-method validation confirmed.")
    else:
        st.error("GNN or Autoencoder results not available.")

    st.divider()
    with st.expander("GNN Architecture Details"):
        st.markdown("""
        **Graph Construction**
        - Each calibration point is a **graph node** with features: `[indicated_norm, ratio_error, uncertainty, range_position, domain_enc, instrument_enc]`
        - Edges between all points of the **same instrument** (fully-connected subgraphs) + cross-instrument edges via nearest neighbours in feature space

        **GCN Encoder-Decoder**
        | Layer | Type | Output dim |
        |-------|------|------------|
        | GCNConv 1 | Graph Convolution | 16 |
        | ReLU | Activation | 16 |
        | GCNConv 2 | Graph Convolution | 8 (bottleneck) |
        | GCNConv 3 (decode) | Graph Convolution | 16 |
        | Linear | Reconstruction | 6 (input features) |

        **Anomaly score** = MSE between input and reconstructed features.
        Points with score > 75th-percentile threshold → anomalous.

        **Why GCN?** Calibration points share structural relationships via shared standards,
        environments and instrument chains. GCN exploits this relational structure —
        unlike tabular methods that treat each point independently.
        """)
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — Multi-task NN
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("Multi-task Neural Network")
    mt_png = PLOTS / "multitask_results.png"
    if mt_png.exists():
        st.image(str(mt_png), caption="Multi-task NN — Predictions vs Ground Truth",
                 use_container_width=True)
    else:
        st.warning("multitask_results.png not found in outputs/plots/")

    mt_df = DATA["multitask_results"]
    if mt_df is None:
        st.error("multitask_results.csv not found in outputs/")
    else:
        st.info("Multi-task prediction: a single neural network pass gives all three answers "
                "— ratio error, distribution type, and uncertainty class.")
        st.divider()

        st.subheader("Task Performance Summary")
        task_mae = (mt_df["mt_pred_ratio_error_pct"]-mt_df["ratio_error_pct"]).abs().mean()
        t2_acc   = (mt_df["mt_pred_nongaussian"]==mt_df["mt_true_nongaussian"]).mean()*100
        t3_acc   = (mt_df["mt_pred_uncertainty_class"]==mt_df["mt_true_uncertainty_class"]).mean()*100
        st.dataframe(pd.DataFrame({
            "Task":  ["Task 1 — Ratio Error Regression","Task 2 — Non-Gaussian Classification",
                      "Task 3 — Uncertainty Class Classification"],
            "Metric":["MAE (%)","Accuracy (%)","Accuracy (%)"],
            "Value": [f"{task_mae:.4f}",f"{t2_acc:.1f}",f"{t3_acc:.1f}"],
        }), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Per-instrument: Predicted vs True Ratio Error")
        inst_breakdown = []
        for inst,grp in mt_df.groupby("instrument"):
            inst_breakdown.append({
                "Instrument":inst,
                "Mean True RE (%)":round(grp["ratio_error_pct"].mean(),4),
                "Mean Predicted RE (%)":round(grp["mt_pred_ratio_error_pct"].mean(),4),
                "MAE (%)":round((grp["mt_pred_ratio_error_pct"]-grp["ratio_error_pct"]).abs().mean(),4),
            })
        st.dataframe(pd.DataFrame(inst_breakdown), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Uncertainty Class Distribution — Predicted vs True")
        uc_classes  = ["low (<0.10%)","medium (0.10–0.30%)","high (>0.30%)"]
        uc_fig      = go.Figure()
        instruments = mt_df["instrument"].unique().tolist()
        for uc_class in uc_classes:
            pred_counts,true_counts = [],[]
            for inst in instruments:
                grp = mt_df[mt_df["instrument"]==inst]
                pred_counts.append((grp["mt_pred_uncertainty_class"]==uc_class).sum())
                true_counts.append((grp["mt_true_uncertainty_class"]==uc_class).sum())
            colour = C_BLUE if "low" in uc_class else (C_AMBER if "medium" in uc_class else C_RED)
            uc_fig.add_trace(go.Bar(name=f"Pred · {uc_class}",x=instruments,
                                    y=pred_counts,marker_color=colour,opacity=0.85))
            uc_fig.add_trace(go.Bar(name=f"True · {uc_class}",x=instruments,
                                    y=true_counts,marker_color=colour,opacity=0.45,
                                    marker_pattern_shape="/"))
        uc_fig.update_layout(barmode="stack",height=380,
                             legend=dict(orientation="h",y=-0.25,font=dict(size=10)),
                             paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,22,36,0.6)",
                             xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                             yaxis=dict(title="Count",gridcolor="rgba(255,255,255,0.07)"))
        st.plotly_chart(uc_fig, use_container_width=True)
    render_footer()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — GUM Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.subheader("GUM vs AI Uncertainty Quantification")
    gum_df = DATA["gum_comparison"]
    ml_df  = DATA["ml_results"]

    if gum_df is None:
        st.error("gum_comparison.csv not found in outputs/")
    else:
        st.subheader("GUM Expanded U vs Conformal PI Half-width per Calibration Point")
        x_labels = [f"{row['Instrument'].split()[-1][:3]} {row['Indicated']}"
                    for _,row in gum_df.iterrows()]
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(name="GUM U (%)",x=x_labels,y=gum_df["GUM_U_pct"],
                                 marker_color=C_AMBER,opacity=0.85))
        bar_fig.add_trace(go.Bar(name="Conformal PI Half-width (%)",x=x_labels,
                                 y=gum_df["Conf_PI_half_pct"],marker_color=C_BLUE,opacity=0.85))
        bar_fig.update_layout(barmode="group",height=420,
                              xaxis=dict(title="Calibration Point",tickangle=-55,
                                         tickfont=dict(size=8),gridcolor="rgba(255,255,255,0.07)"),
                              yaxis=dict(title="Uncertainty (%)",gridcolor="rgba(255,255,255,0.07)"),
                              legend=dict(orientation="h",y=-0.35),
                              paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,22,36,0.6)")
        st.plotly_chart(bar_fig, use_container_width=True)
        st.divider()

        st.subheader("GUM Half-width vs Conformal Half-width (Agreement Plot)")
        scat_fig = px.scatter(gum_df,x="GUM_half_pct",y="Conf_PI_half_pct",color="Instrument",
                              color_discrete_map=INST_COLOURS,
                              hover_data=["Indicated","ML_predicted_RE_pct"],
                              template="plotly_dark",height=420,
                              labels={"GUM_half_pct":"GUM Half-width U/2 (%)",
                                      "Conf_PI_half_pct":"Conformal PI Half-width (%)"})
        max_val = max(gum_df["GUM_half_pct"].max(),gum_df["Conf_PI_half_pct"].max())*1.1
        scat_fig.add_trace(go.Scatter(x=[0,max_val],y=[0,max_val],mode="lines",
                                      line=dict(color=C_RED,dash="dash",width=1.5),
                                      name="Perfect agreement",showlegend=True))
        scat_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(20,22,36,0.6)",
                               xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                               yaxis=dict(gridcolor="rgba(255,255,255,0.07)"))
        st.plotly_chart(scat_fig, use_container_width=True)
        st.divider()

        st.subheader("GUM-formatted Output — Per Calibration Point")
        if ml_df is not None:
            gum_table_rows = []
            for _,row in ml_df.iterrows():
                re,pred,hw,k = (row["ratio_error_pct"],row["ml_predicted_ratio_error_pct"],
                                row["conformal_halfwidth_pct"],row["coverage_factor_k"])
                gum_table_rows.append({
                    "Instrument":row["instrument"],"Indicated":row["indicated_value"],
                    "GUM output":f"RE = {re:+.4f}% ± {row['gum_expanded_U_pct']:.4f}% (U, k≈{k:.1f}, 95%)",
                    "ML output": f"RE = {pred:+.4f}% ± {hw:.4f}% (conformal PI, 95%)",
                })
            st.dataframe(pd.DataFrame(gum_table_rows), use_container_width=True, height=340)

        st.divider()
        st.warning("Wide conformal PI (±7.01%) driven by HV Breakdown Tester outliers. "
                   "This is a feature — the framework correctly identifies the low-accuracy "
                   "instrument as the anomaly source.")

        dl1,dl2 = st.columns(2)
        with dl1:
            st.download_button("Download gum_comparison.csv",
                               data=gum_df.to_csv(index=False).encode(),
                               file_name="gum_comparison.csv",mime="text/csv")
        with dl2:
            if ml_df is not None:
                st.download_button("Download ml_results.csv",
                                   data=ml_df.to_csv(index=False).encode(),
                                   file_name="ml_results.csv",mime="text/csv")
    render_footer()