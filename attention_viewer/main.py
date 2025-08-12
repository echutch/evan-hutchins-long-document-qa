from torch import cuda, tensor
import json
import numpy as np
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from typing import Optional
from dotenv import load_dotenv
import os
from doc_loader import load_and_split_pdf

# Load environment variables from .env file
load_dotenv()

# TODO: Get rid of layers, only save final layer? Or add option to save all layers?
# TODO: Add option to save all heads or only average over heads?

# TODO: Document parsing setup & allow custom paragraph delimiters

# TODO: Add CLI arguments for model name, output path, device, default layer, and top-N paragraphs
# TODO: README with usage instructions
# TODO: Add logging for errors and progress

def generate_light_attention_viewer(
    file_name: str,
    question: str,
    model_name: str = "EleutherAI/pythia-410m",
    output_html: str = "attention_visual/light_attention_viewer.html",
    device: Optional[str] = None,
    default_layer: int = 0,
    default_top_n: int = 5
):
    """
    Generate a lightweight HTML attention viewer showing top-N paragraph highlights.

    Attention scores are averaged over all heads per layer.
    No token zoom, no head selector.

    Args:
        document: Full document text with paragraphs separated by blank lines.
        question: Query appended to the prompt.
        model_name: HuggingFace model to load.
        output_html: Path to save the HTML file.
        device: 'cuda', 'cpu' or None to auto-select.
        default_layer: Default layer index for UI.
        default_top_n: Default top-N paragraphs to highlight.
    """

    if device is None:
        device = "cuda" if cuda.is_available() else "cpu"
    
    print(f"Initializing model '{model_name}' on device '{device}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    print("Model loaded successfully")

    print("Splitting document into paragraphs...")
    document = load_and_split_pdf(file_name)

    paragraphs = [p.strip() for p in document if p.strip()]
    print(f"Processing document with {len(paragraphs)} paragraphs...")
    prompt_prefix = "Document:\n"
    prompt_tokens = tokenizer.encode(prompt_prefix, add_special_tokens=False)

    para_token_spans = []
    current_pos = len(prompt_tokens)

    for para in paragraphs:
        ids = tokenizer.encode(para + "\n\n", add_special_tokens=False)
        para_token_spans.append((current_pos, current_pos + len(ids)))
        prompt_tokens.extend(ids)
        current_pos += len(ids)

    question_prefix = f"Prompt: {question}\nAnswer:"
    qids = tokenizer.encode(question_prefix, add_special_tokens=False)
    prompt_tokens.extend(qids)

    tokens = tensor([prompt_tokens]).to(device)
    answer_start_idx = len(prompt_tokens) - len(qids)
    
    print(f"Running model inference with {len(prompt_tokens)} tokens...")
    logits, cache = model.run_with_cache(tokens)
    print("Model inference completed")

    answer_tokens = logits[0, answer_start_idx:].argmax(dim=-1)
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    n_layers = model.cfg.n_layers
    print(f"Processing attention patterns across {n_layers} layers...")

    layer_para_scores = []

    for layer in range(n_layers):
        attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [heads, q_len, k_len]
        doc_token_idxs = list(range(answer_start_idx))
        answer_token_idxs = list(range(answer_start_idx, tokens.shape[1]))

        # Get mean attention over heads and answer tokens for each doc token
        mean_attn = attn[:, answer_token_idxs][:, :, doc_token_idxs].mean(dim=0).mean(dim=0).detach().cpu().numpy()

        para_scores = []
        for (s, e) in para_token_spans:
            token_scores = mean_attn[s:e]
            para_scores.append(float(token_scores.mean()) if token_scores.size > 0 else 0.0)
        layer_para_scores.append(para_scores)

    print("Preparing data and generating HTML viewer...")
    json_data = {
        "meta": {
            "n_layers": n_layers,
            "paragraph_count": len(paragraphs),
            "model_name": model_name,
            "question": question,
            "answer": answer,
        },
        "paragraphs": paragraphs,
        "layer_para_scores": layer_para_scores
    }

    json_blob = json.dumps(json_data)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Light Attention Viewer</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #fff; color: #222; }}
  .controls {{ margin-bottom: 15px; }}
  select, input[type=number] {{ margin-right: 10px; padding: 4px; font-size: 1em; }}
  .para {{ margin-bottom: 10px; padding: 8px; border-radius: 6px; cursor: default; }}
  .dimmed {{ color: #999; background-color: #f5f5f5; }}
</style>
</head>
<body>

<div class="qa-section">
  <h2>Question</h2>
  <p>{question}</p>
  <h2>LLM Response</h2>
  <p>{answer.strip()}</p>
</div>

<h2>Attention Visualization</h2>

<div class="controls">
  <label>Layer:
    <select id="layerSelect"></select>
  </label>
  <label>Top N:
    <input type="number" id="topNInput" min="1" max="{len(paragraphs)}" value="{default_top_n}" />
  </label>
</div>

<div id="docContainer"></div>

<script>
const DATA = {json_blob};

const layerSelect = document.getElementById('layerSelect');
const topNInput = document.getElementById('topNInput');
const docContainer = document.getElementById('docContainer');

for(let i=0; i<DATA.meta.n_layers; i++){{
  let opt = document.createElement('option');
  opt.value = i;
  opt.textContent = 'Layer ' + i;
  layerSelect.appendChild(opt);
}}

layerSelect.value = {default_layer};

function scoreToColor(norm) {{
  norm = Math.min(Math.max(norm, 0), 1);
  let r1=255, g1=255, b1=204;
  let r2=177, g2=0, b2=38;
  let r = Math.round(r1 + (r2-r1)*norm);
  let g = Math.round(g1 + (g2-g1)*norm);
  let b = Math.round(b1 + (b2-b1)*norm);
  return "rgb(" + r + "," + g + "," + b + ")";
}}

function render() {{
  docContainer.innerHTML = '';
  const layer = parseInt(layerSelect.value);
  let topN = parseInt(topNInput.value);
  if(isNaN(topN) || topN < 1) topN = 1;
  if(topN > DATA.paragraphs.length) topN = DATA.paragraphs.length;
  topNInput.value = topN;

  const scores = DATA.layer_para_scores[layer];
  const maxScore = Math.max(...scores, 1e-12);

  const sortedIndices = [...scores.keys()].sort((a,b) => scores[b] - scores[a]);
  const topIndices = new Set(sortedIndices.slice(0, topN));

  for(let i=0; i<DATA.paragraphs.length; i++) {{
    const p = document.createElement('div');
    p.className = 'para';
    if(topIndices.has(i)) {{
      const normScore = scores[i]/maxScore;
      p.style.backgroundColor = scoreToColor(normScore);
      p.title = 'Paragraph ' + i + ' — score: ' + scores[i].toFixed(6);
    }} else {{
      p.classList.add('dimmed');
      p.title = 'Paragraph ' + i + ' — outside top ' + topN;
    }}
    p.textContent = DATA.paragraphs[i];
    docContainer.appendChild(p);
  }}
}}

layerSelect.addEventListener('change', render);
topNInput.addEventListener('input', render);

render();
</script>

</body>
</html>
"""

    print(f"Writing HTML file to '{output_html}'...")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Light attention viewer saved to {output_html}")
    print("Open it in a browser. Use controls to pick layer and top-N paragraphs.")



if __name__ == "__main__":
    # Example usage:
    # your_long_document_text = """
    # [PASTE YOUR LONG DOCUMENT HERE, WITH PARAGRAPHS SEPARATED BY BLANK LINES]

    # [SECTION 1] Berlin is the capital of Germany.

    # [SECTION 2] Madrid is the capital of Spain.

    # [SECTION 3] Paris is the capital of France. It is known for the Eiffel Tower.

    # [SECTION 4] Rome is the capital of Italy.

    # """

    file_name = "./test_docs/PaperQA_single.pdf"  # Path to your PDF file

    question = """Answer in a direct and concise tone, I am in a hurry. Your audience is an expert, so be
  highly specific. If there are ambiguous terms or acronyms, first define them.
  Write an answer with five sentences maximum for the question below based on the provided context.
  If the context provides insufficient information, reply ''I cannot answer''. Answer in an unbiased, comprehensive,
  and scholarly tone. If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences.

  Question: How does PaperQA compare to expert humans?
  """

    generate_light_attention_viewer(
        file_name=file_name,
        question=question,
        model_name='meta-llama/Llama-3.2-1B-Instruct',
        default_layer=15,
        default_top_n=2,
        device='cpu',
    )



