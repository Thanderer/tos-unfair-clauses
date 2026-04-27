import gradio as gr
import requests

from src.data.utils_pdf_text import pdf_to_text
from src.inference.preprocess_input import load_text_input


# =========================
# PREPROCESS
# =========================
def handle_upload(file):
    if file is None:
        return "❌ No file", []

    try:
        if file.name.endswith(".pdf"):
            raw = pdf_to_text(file.name)
        else:
            with open(file.name, "r", encoding="utf-8") as f:
                raw = f.read()

        clauses = load_text_input(raw)

        return f"✅ Ready ({len(clauses)} clauses)", clauses

    except Exception as e:
        return f"❌ Error: {str(e)}", []


def handle_paste(text):
    if not text:
        return "❌ No text", []

    clauses = load_text_input(text)

    return f"✅ Ready ({len(clauses)} clauses)", clauses


# =========================
# BUILD CARDS
# =========================
def build_cards(results, filter_value):

    if filter_value != "ALL":
        results = [r for r in results if r["severity_band"] == filter_value]

    cards_html = ""

    for r in results:
        color = (
            "#7c3aed" if r["severity_band"] == "CRITICAL"
            else "#ef4444" if r["severity_band"] == "HIGH"
            else "#f59e0b" if r["severity_band"] == "MEDIUM"
            else "#10b981"
        )

        cards_html += f"""
        <div style="background:#0f172a;color:white;padding:20px;border-radius:12px;margin-top:15px;border-left:6px solid {color}">
            <span style="background:{color};padding:5px 10px;border-radius:12px;font-size:12px">
                {r['severity_band']}
            </span>

            <p style="margin-top:10px">{r['text']}</p>

            <p style="font-size:13px;color:#cbd5f5">
                💡 {r.get('explanation', '')}
            </p>

            <p style="font-size:12px;color:#94a3b8">
                Score: {r.get('severity_score', '')}/10
            </p>
        </div>
        """

    return cards_html


# =========================
# CALL API
# =========================
def call_api(clauses):
    if not clauses:
        return "<p style='color:red'>No data</p>", "", []

    try:
        res = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"clauses": clauses},
            timeout=120
        )

        data = res.json()
        results = data.get("results", [])

        if not results:
            return "<p style='color:red'>No results</p>", "", []

        # Counts
        critical = sum(1 for r in results if r["severity_band"] == "CRITICAL")
        high = sum(1 for r in results if r["severity_band"] == "HIGH")
        medium = sum(1 for r in results if r["severity_band"] == "MEDIUM")
        safe = sum(1 for r in results if r["severity_band"] == "SAFE")

        summary_html = f"""
        <div style="display:flex;gap:20px;margin-top:20px">

            <div style="flex:1;background:#7c3aed;color:white;padding:20px;border-radius:12px;text-align:center">
                <h1>{critical}</h1><p>Critical</p>
            </div>

            <div style="flex:1;background:#ef4444;color:white;padding:20px;border-radius:12px;text-align:center">
                <h1>{high}</h1><p>High Risk</p>
            </div>

            <div style="flex:1;background:#f59e0b;color:white;padding:20px;border-radius:12px;text-align:center">
                <h1>{medium}</h1><p>Medium</p>
            </div>

            <div style="flex:1;background:#10b981;color:white;padding:20px;border-radius:12px;text-align:center">
                <h1>{safe}</h1><p>Safe</p>
            </div>

        </div>
        """

        cards_html = build_cards(results, "ALL")

        return summary_html, cards_html, results

    except Exception as e:
        return f"<p style='color:red'>API Error: {str(e)}</p>", "", []


# =========================
# FILTER
# =========================
def apply_filter(filter_value, results):
    if not results:
        return ""

    return build_cards(results, filter_value)


# =========================
# UI
# =========================
with gr.Blocks(title="ToS Risk Analyzer") as demo:

    gr.Markdown("## 🚀 ToS Risk Analyzer")

    state = gr.State([])
    results_state = gr.State([])

    with gr.Row():
        file = gr.File(label="Upload ToS (.txt / .pdf)")
        text = gr.Textbox(label="Paste Text", lines=10)

    status = gr.Textbox(label="Status")

    file.change(handle_upload, inputs=file, outputs=[status, state])
    text.change(handle_paste, inputs=text, outputs=[status, state]).then() #only fire ehrn user stops typing

    analyze_btn = gr.Button("✨ Analyze")

    summary = gr.HTML()

    filter_radio = gr.Radio(
        choices=["ALL", "CRITICAL", "HIGH", "MEDIUM", "SAFE"],
        value="ALL",
        label="Filter Results"
    )

    cards = gr.HTML()

    analyze_btn.click(
        fn=call_api,
        inputs=[state],
        outputs=[summary, cards, results_state],
        show_progress=True
    )

    filter_radio.change(
        fn=apply_filter,
        inputs=[filter_radio, results_state],
        outputs=[cards]
    )

demo.launch()
