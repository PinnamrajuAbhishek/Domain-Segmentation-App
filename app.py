from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# ============================================
# CONFIG
# ============================================
DOMAIN_LABEL_COL = "Domain"
CLEAN_COL = "clean_text"
EXCLUDED_COL = "excluded_text"

EXCEL_PATH = r"D:\Domain segmentation\Presentation How to Buy.xlsx"
EXCEL_SHEET = "Updated_how2buy"

SBERT_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # small local LLM

TOP_K_SBERT = 10          # SBERT candidates passed to LLM
MAX_NEW_TOKENS = 160      # limit LLM output length

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================================
# LOAD DOMAIN DATA
# ============================================
df = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET)
df[EXCLUDED_COL] = df[EXCLUDED_COL].fillna("")

# ============================================
# LOAD MODELS
# ============================================
print("Loading SBERT...")
sbert_model = SentenceTransformer(SBERT_MODEL, device=device)

print("Loading Qwen2.5-0.5B-Instruct (local LLM)...")
llm_tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL,
    trust_remote_code=True
)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)
llm_model.eval()

# Pre-encode domain embeddings once
print("Encoding domain clean & excluded text...")
clean_emb = sbert_model.encode(
    df[CLEAN_COL].tolist(),
    convert_to_tensor=True,
    device=device
)
excl_emb = sbert_model.encode(
    df[EXCLUDED_COL].tolist(),
    convert_to_tensor=True,
    device=device
)

# ============================================
# LLM PROMPT BUILDER
# ============================================
def build_llm_prompt(query, candidate_domains):
    """
    candidate_domains: list of dicts from df for SBERT top-K
    """
    lines = []
    for i, d in enumerate(candidate_domains, 1):
        label = d[DOMAIN_LABEL_COL]
        clean = (d[CLEAN_COL] or "").strip()
        excl = (d[EXCLUDED_COL] or "").strip()
        lines.append(
            f"{i}. label: {label}\n"
            f"   clean: {clean}\n"
            f"   excluded: {excl}"
        )
    domain_block = "\n".join(lines)

    prompt = f"""
You are a domain classification assistant.

You get:
- A user query.
- A list of candidate domain labels with descriptions.

Your task:
1. Choose the BEST 3 domain labels that match the user query.
2. IMPORTANT: Each label must be EXACTLY one of the labels from the candidate list.
3. Do NOT invent generic names like "label1" or "label2".
4. Output ONLY valid JSON in this exact format:

{{
  "top3": ["label_A", "label_B", "label_C"]
}}

User query:
\"\"\"{query}\"\"\"

Candidate domains:
{domain_block}

Now return ONLY the JSON.
"""
    return prompt.strip()

# ============================================
# CALL LOCAL LLM (Qwen)
# ============================================
def call_llm(prompt: str) -> str:
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return llm_tokenizer.decode(gen_ids, skip_special_tokens=True)

# ============================================
# SBERT + LLM CLASSIFICATION FOR ONE QUERY
# ============================================
def classify_single_query(query: str):
    """
    1) SBERT score = sim(clean) - sim(excluded) over all domains
    2) Take SBERT top-K domains
    3) Ask Qwen to pick best 3 labels from that list
    4) Fallback to SBERT top-3 if LLM fails
    """
    # Encode query
    q_emb = sbert_model.encode(query, convert_to_tensor=True, device=device)

    # SBERT scores
    sim_clean = util.cos_sim(q_emb, clean_emb)[0]
    sim_excl  = util.cos_sim(q_emb, excl_emb)[0]
    final_scores = sim_clean - sim_excl

    # SBERT Top-K
    top_scores, top_idx = torch.topk(final_scores, k=min(TOP_K_SBERT, len(df)))
    candidate_domains = df.iloc[top_idx.cpu().tolist()].to_dict(orient="records")

    # LLM Reranking
    prompt = build_llm_prompt(query, candidate_domains)
    llm_output = call_llm(prompt)

    # Try to parse JSON from LLM
    try:
        s = llm_output.find("{")
        e = llm_output.rfind("}") + 1
        parsed = json.loads(llm_output[s:e])
        labels = parsed["top3"]

        # Make sure we got 3 labels and they exist in candidate list
        if not isinstance(labels, list) or len(labels) < 3:
            raise ValueError("top3 not a list of length 3")

        candidate_labels = {c[DOMAIN_LABEL_COL] for c in candidate_domains}
        filtered = [lab for lab in labels if lab in candidate_labels]

        if len(filtered) < 3:
            raise ValueError("Some labels not in candidate list")

        top3 = filtered[:3]

    except Exception as e:
        # Fallback: SBERT top-3
        print("LLM parse/validation failed, falling back to SBERT top-3. Error:", e)
        top3 = [
            candidate_domains[0][DOMAIN_LABEL_COL],
            candidate_domains[1][DOMAIN_LABEL_COL],
            candidate_domains[2][DOMAIN_LABEL_COL],
        ]

    return top3

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/segment", methods=["POST"])
def segment_domain():
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Query is empty"}), 400

    print("Query received:", user_query)

    top3 = classify_single_query(user_query)

    response_str = (
        "Top 3 predicted domains:\n"
        f"1. {top3[0]}\n"
        f"2. {top3[1]}\n"
        f"3. {top3[2]}\n"
    )

    return jsonify({"segmentation": response_str})

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    print("Flask server running at http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
