# src/frontend/app.py

import gradio as gr
import random
from src.inference.preprocess_input import load_text_input
from src.data.utils_pdf_text import pdf_to_text

# ---------------- MOCK ANALYSIS ----------------
def mock_analyze(clauses):
    results = []
    for c in clauses:
        label = random.choices(
            ["HIGH", "MEDIUM", "SAFE"],
            weights=[0.3, 0.3, 0.4]
        )[0]

        results.append({
            "text": c["text"],
            "label": label
        })
    return results


# ---------------- PREPROCESS ----------------
def handle_upload(file):
    if file is None:
        return "❌ No file uploaded", None

    try:
        if file.name.endswith(".pdf"):
            raw_text = pdf_to_text(file.name)
        else:
            with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

        clauses = load_text_input(raw_text)

        payload = {"clauses": clauses}

        print("✅ PREPROCESS DONE:", len(clauses))

        return f"✅ Ready ({len(clauses)} clauses)", payload

    except Exception as e:
        return f"❌ Error: {str(e)}", None


# ---------------- RENDER ----------------
def render_cards(data):
    def card(r):
        cls = r["label"].lower()
        return f"""
        <div class="clause {cls}">
            <span class="tag">{r['label']}</span>
            <p>{r['text']}</p>
        </div>
        """
    return "".join([card(r) for r in data])


# ---------------- ANALYZE ----------------
def analyze(payload):
    if payload is None:
        return "❌ No data", "", "", []

    clauses = payload["clauses"]
    results = mock_analyze(clauses)

    high = [r for r in results if r["label"] == "HIGH"]
    medium = [r for r in results if r["label"] == "MEDIUM"]
    safe = [r for r in results if r["label"] == "SAFE"]

    summary_html = f"""
    <div class="summary-container">
        <div class="card high">{len(high)}<span>High Risk</span></div>
        <div class="card medium">{len(medium)}<span>Medium</span></div>
        <div class="card safe">{len(safe)}<span>Safe</span></div>
    </div>
    """

    html = render_cards(results)

    return "✅ Analysis Complete", summary_html, html, results


# ---------------- FILTER ----------------
def apply_filter(results, filter_type):
    if not results:
        return ""

    if filter_type == "ALL":
        filtered = results
    else:
        filtered = [r for r in results if r["label"] == filter_type]

    return render_cards(filtered)


# ---------------- UI ----------------
with gr.Blocks() as demo:

    gr.Markdown("# 🚀 ToS Risk Analyzer")

    payload_state = gr.State()
    results_state = gr.State()

    with gr.Row():
        file_input = gr.File(label="Upload ToS")
        text_input = gr.Textbox(label="Paste Text", lines=8)

    status = gr.Textbox(label="Status", interactive=False)

    filter_radio = gr.Radio(
        ["ALL", "HIGH", "MEDIUM", "SAFE"],
        value="ALL",
        label="Filter Clauses"
    )

    analyze_btn = gr.Button("✨ Analyze")

    summary = gr.HTML()
    output = gr.HTML()

    # Upload → preprocess
    file_input.change(
        handle_upload,
        inputs=file_input,
        outputs=[status, payload_state]
    )

    # Analyze → generate results
    analyze_btn.click(
        analyze,
        inputs=payload_state,
        outputs=[status, summary, output, results_state]
    )

    # Filter → instant update
    filter_radio.change(
        apply_filter,
        inputs=[results_state, filter_radio],
        outputs=output
    )


# ---------------- PREMIUM CSS ----------------
demo.launch(
    css="""
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

h1 {
    font-size: 28px;
    font-weight: 700;
}

/* SUMMARY */
.summary-container {
    display: flex;
    gap: 20px;
    margin: 25px 0;
}

.card {
    flex: 1;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    color: white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

.card span {
    display: block;
    margin-top: 6px;
    font-size: 14px;
}

.high { background: linear-gradient(135deg, #ef4444, #dc2626); }
.medium { background: linear-gradient(135deg, #f59e0b, #fbbf24); }
.safe { background: linear-gradient(135deg, #10b981, #22c55e); }

/* CLAUSE */
.clause {
    background: #1e293b;
    padding: 18px;
    margin: 14px 0;
    border-radius: 12px;
    border-left: 6px solid transparent;
    transition: 0.25s;
}

.clause:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

.clause.high { border-left-color: #ef4444; }
.clause.medium { border-left-color: #f59e0b; }
.clause.safe { border-left-color: #22c55e; }

.tag {
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 8px;
    color: white;
}

.clause.high .tag { background: #ef4444; }
.clause.medium .tag { background: #f59e0b; }
.clause.safe .tag { background: #22c55e; }

.clause p {
    font-size: 14px;
    line-height: 1.6;
    color: #e2e8f0;
}

/* RADIO */
.gr-radio {
    display: flex !important;
    gap: 10px;
}

.gr-radio label {
    flex: 1;
    text-align: center;
    padding: 10px;
    border-radius: 10px;
    background: #1e293b;
    cursor: pointer;
    transition: 0.3s;
}

.gr-radio label:hover {
    background: #334155;
}

.gr-radio input[type="radio"] {
    display: none;
}

.gr-radio input[type="radio"]:checked + span {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 10px;
    border-radius: 10px;
}
"""
)