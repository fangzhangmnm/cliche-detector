"""
Cliché Detector — measure token-level entropy of text using a language model.
Outputs a single HTML file with token/sentence view toggle.
"""

import argparse
import math
import os
import re
import html as html_lib

SENTENCE_ENDERS = set("。！？.!?\n")


# ---------------------------------------------------------------------------
# Analyze — compute logprobs in memory
# ---------------------------------------------------------------------------

def analyze(text, model_id, top_k, chunk_size=4096):
    """Run LLM inference on text, return (model_id, top_k, tokens_data)."""
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading model {model_id} on {device} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, local_files_only=True)
    except OSError:
        print("Model not cached locally, downloading...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()

    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    num_tokens = input_ids.shape[1]
    print(f"Text: {len(text)} chars, {num_tokens} tokens")

    first_id = input_ids[0, 0].item()
    vocab_size = len(tokenizer)
    tokens_data = [{
        "token": tokenizer.decode([first_id]),
        "token_id": first_id,
        "logp": round(-math.log(vocab_size), 4),
        "top": [],
    }]
    past_key_values = None

    pbar = tqdm(total=num_tokens - 1, desc="Processing tokens")
    pos = 0
    while pos < num_tokens:
        chunk_end = min(pos + chunk_size, num_tokens)
        chunk_ids = input_ids[:, pos:chunk_end]

        with torch.no_grad():
            outputs = model(chunk_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        chunk_logits = outputs.logits[0]

        for j in range(chunk_logits.shape[0]):
            next_pos = pos + j + 1
            if next_pos >= num_tokens:
                break

            token_id = input_ids[0, next_pos].item()
            log_probs = torch.log_softmax(chunk_logits[j], dim=-1)
            token_logp = log_probs[token_id].item()
            top_vals, top_ids = torch.topk(log_probs, top_k)

            tokens_data.append({
                "token": tokenizer.decode([token_id]),
                "token_id": token_id,
                "logp": round(token_logp, 4),
                "top": [
                    {
                        "token": tokenizer.decode([top_ids[k].item()]),
                        "token_id": top_ids[k].item(),
                        "logp": round(top_vals[k].item(), 4),
                    }
                    for k in range(top_k)
                ],
            })
            pbar.update(1)

        pos = chunk_end
    pbar.close()

    return tokens_data


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def bits_to_color(bits, min_bits=0, max_bits=16):
    if max_bits <= min_bits:
        t = 0.5
    else:
        t = max(0, min((bits - min_bits) / (max_bits - min_bits), 1.0))
    if t < 0.5:
        r = int(255 * (t * 2))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (t - 0.5) * 2))
    return f"rgba({r},{g},0,0.5)"


def split_paragraphs(tokens, bits_list):
    full_text = "".join(t["token"] for t in tokens)
    char_to_tok = []
    for i, t in enumerate(tokens):
        char_to_tok.extend([i] * len(t["token"]))

    para_breaks = [m.span() for m in re.finditer(r'\n\s*\n', full_text)]

    paragraphs = []
    prev_char = 0
    for break_start, break_end in para_breaks:
        if break_start <= prev_char:
            continue
        tok_start = char_to_tok[prev_char] if prev_char < len(char_to_tok) else len(tokens)
        tok_end = char_to_tok[break_start - 1] + 1 if break_start > 0 and break_start - 1 < len(char_to_tok) else len(tokens)
        if tok_end > tok_start:
            p_bits = bits_list[tok_start:tok_end]
            paragraphs.append({
                "tokens": tokens[tok_start:tok_end],
                "bits": p_bits,
                "total_bits": sum(p_bits),
                "avg_bits": sum(p_bits) / len(p_bits),
                "n_tokens": len(p_bits),
            })
        prev_char = break_end

    if prev_char < len(full_text):
        tok_start = char_to_tok[prev_char]
        p_bits = bits_list[tok_start:]
        p_text = full_text[prev_char:]
        if p_bits and p_text.strip():
            paragraphs.append({
                "tokens": tokens[tok_start:],
                "bits": p_bits,
                "total_bits": sum(p_bits),
                "avg_bits": sum(p_bits) / len(p_bits),
                "n_tokens": len(p_bits),
            })
    return paragraphs


def split_sentences(tokens, bits):
    """Split tokens+bits into sentence groups. Returns list of (tokens, bits) tuples."""
    sentences = []
    cur_toks, cur_bits = [], []
    for t, b in zip(tokens, bits):
        cur_toks.append(t)
        cur_bits.append(b)
        if any(c in SENTENCE_ENDERS for c in t["token"]):
            text = "".join(tk["token"] for tk in cur_toks)
            if text.strip():
                sentences.append((cur_toks, cur_bits))
                cur_toks, cur_bits = [], []
    if cur_bits:
        text = "".join(tk["token"] for tk in cur_toks)
        if text.strip() and sentences:
            prev_toks, prev_bits = sentences[-1]
            sentences[-1] = (prev_toks + cur_toks, prev_bits + cur_bits)
        elif text.strip():
            sentences.append((cur_toks, cur_bits))
    return sentences


def generate_html(tokens, bits_list, paragraphs, stats_text, title="Cliché Detector", max_bits=16):
    avg_bits = sum(bits_list) / len(bits_list)
    stats_escaped = html_lib.escape(stats_text)

    # Compute sentence-level color ranges
    all_sent_avgs = []
    all_para_avgs = [p["avg_bits"] for p in paragraphs]
    for p in paragraphs:
        for s_toks, s_bits in split_sentences(p["tokens"], p["bits"]):
            all_sent_avgs.append(sum(s_bits) / len(s_bits))

    p_mu = sum(all_para_avgs) / len(all_para_avgs)
    p_sigma = (sum((x - p_mu) ** 2 for x in all_para_avgs) / len(all_para_avgs)) ** 0.5
    p_min = max(0, p_mu - 2 * p_sigma)
    p_max = p_mu + 2 * p_sigma

    if all_sent_avgs:
        s_mu = sum(all_sent_avgs) / len(all_sent_avgs)
        s_sigma = (sum((x - s_mu) ** 2 for x in all_sent_avgs) / len(all_sent_avgs)) ** 0.5
    else:
        s_mu, s_sigma = avg_bits, 1.0
    s_min = max(0, s_mu - 3 * s_sigma)
    s_max = s_mu + 3 * s_sigma

    # Build single DOM: para > sent > tok
    body_parts = []
    for p in paragraphs:
        p_color = bits_to_color(p["avg_bits"], min_bits=p_min, max_bits=p_max)
        p_tip = f"para: {p['avg_bits']:.2f} bit/tok, {p['n_tokens']} tok, {p['total_bits']:.0f} bits"
        body_parts.append(
            f'<div class="para" data-bits="{p["avg_bits"]:.2f}" '
            f'data-color-sent="{p_color}">'
            f'<span class="para-tip">{html_lib.escape(p_tip)}</span>'
        )
        for s_toks, s_bits in split_sentences(p["tokens"], p["bits"]):
            s_avg = sum(s_bits) / len(s_bits)
            s_total = sum(s_bits)
            s_color = bits_to_color(s_avg, min_bits=s_min, max_bits=s_max)
            s_tip = f"{s_avg:.2f} bit/tok, {len(s_bits)} tok, {s_total:.0f} bits"
            body_parts.append(
                f'<span class="sent" data-bits="{s_avg:.2f}" '
                f'data-color-sent="{s_color}">'
                f'<span class="sent-tip">{html_lib.escape(s_tip)}</span>'
            )
            for t, bits in zip(s_toks, s_bits):
                text = t["token"]
                tok_color = bits_to_color(bits)
                escaped = html_lib.escape(text).replace("\n", "<br>")
                tip_lines = [f"{html_lib.escape(text)}  {bits:.1f} bits", ""]
                for entry in t["top"]:
                    entry_bits = -entry["logp"] / math.log(2)
                    tip_lines.append(f"{html_lib.escape(entry['token']):8s} {entry_bits:5.1f}")
                tok_tip = "\n".join(tip_lines)
                body_parts.append(
                    f'<span class="tok" data-bits="{bits:.2f}" '
                    f'data-color-tok="{tok_color}">'
                    f'{escaped}<span class="tok-tip">{tok_tip}</span></span>'
                )
            body_parts.append('</span>')
        body_parts.append('</div>')

    content = "".join(body_parts)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html_lib.escape(title)}</title>
<style>
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; padding: 20px; max-width: 900px; margin: 0 auto; background: #fff; color: #000; font-size: 14px; line-height: 1.6; }}
.controls {{ position: sticky; top: 0; background: #fff; padding: 6px 0; border-bottom: 1px solid #ddd; margin-bottom: 12px; z-index: 10; }}
.stats {{ font-family: monospace; font-size: 12px; color: #555; margin-bottom: 4px; white-space: pre; line-height: 1.3; }}
.row {{ display: flex; align-items: center; gap: 8px; font-family: monospace; font-size: 12px; color: #555; margin-bottom: 4px; }}
.row input {{ width: 120px; }}
.tabs {{ display: flex; gap: 0; }}
.tab {{ padding: 4px 16px; cursor: pointer; font-family: monospace; font-size: 13px; border: 1px solid #ccc; background: #f5f5f5; color: #555; user-select: none; }}
.tab:first-child {{ border-radius: 4px 0 0 4px; }}
.tab:last-child {{ border-radius: 0 4px 4px 0; }}
.tab.active {{ background: #333; color: #fff; border-color: #333; }}
.legend-bar {{ width: 100px; height: 10px; border-radius: 3px; background: linear-gradient(to right, rgba(0,255,0,0.5), rgba(255,255,0,0.5), rgba(255,0,0,0.5)); }}
.hide {{ display: none; }}

/* Shared structure */
.para {{ padding: 2px 4px; border-radius: 3px; margin-bottom: 4px; display: block; position: relative; border: 1px solid #ddd; }}
.sent {{ padding: 1px 2px; border-radius: 2px; position: relative; border: 1px solid #e8e8e8; }}
.tok {{ padding: 1px 0; border-radius: 2px; cursor: default; position: relative; }}

/* Tooltips — all hidden by default, mode toggles which ones show */
.tok-tip, .sent-tip, .para-tip {{ display: none; position: absolute; top: 100%; left: 0; background: #333; color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 13px; font-family: monospace; white-space: pre; z-index: 100; pointer-events: none; }}

/* Token mode */
.mode-token .tok {{ background: var(--tok-color); margin: 0 1px; }}
.mode-token .tok:hover {{ outline: 2px solid #333; }}
.mode-token .tok:hover .tok-tip {{ display: block; }}
.mode-token .tok.hidden {{ display: none; }}
.mode-token .sent {{ background: transparent; }}
.mode-token .para {{ background: var(--para-color); }}
.mode-token .para:hover > .para-tip {{ display: block; }}

/* Sentence mode */
.mode-sentence .tok {{ background: transparent; margin: 0; }}
.mode-sentence .sent {{ background: var(--sent-color); cursor: default; }}
.mode-sentence .sent:hover {{ outline: 2px solid #333; }}
.mode-sentence .sent:hover > .sent-tip {{ display: block; }}
.mode-sentence .para {{ background: var(--para-color); }}
.mode-sentence .para:hover > .para-tip {{ display: block; }}
.mode-sentence .sent.hidden {{ display: none; }}
</style></head><body>
<div class="controls">
<div class="stats">{stats_escaped}</div>
<div class="row">
  <div class="tabs">
    <div class="tab active" data-mode="token">Token</div>
    <div class="tab" data-mode="sentence">Sentence</div>
  </div>
  <span id="slider-token">hide under <input type="range" id="tok-threshold" min="0" max="{max_bits}" step="0.1" value="0"> <span id="tok-threshold-val">0.0</span> bit</span>
  <span id="slider-sentence" class="hide">hide under <input type="range" id="sent-threshold" min="{s_min:.1f}" max="{s_max:.1f}" step="0.1" value="{s_min:.1f}"> <span id="sent-threshold-val">{s_min:.1f}</span> bit/tok</span>
</div>
<div id="legend-token" class="row">
  <span>token:</span><span>0 bit</span><div class="legend-bar"></div><span>{max_bits}+ bit</span>
  <span>&nbsp; paragraph:</span><span>{p_min:.1f}</span><div class="legend-bar"></div><span>{p_max:.1f}+</span>
</div>
<div id="legend-sentence" class="row hide">
  <span>sentence:</span><span>{s_min:.1f}</span><div class="legend-bar"></div><span>{s_max:.1f}+</span>
  <span>&nbsp; paragraph:</span><span>{p_min:.1f}</span><div class="legend-bar"></div><span>{p_max:.1f}+</span>
</div>
</div>
<div id="content" class="mode-token">{content}</div>
<script>
// Apply CSS custom properties from data attributes
document.querySelectorAll('.tok').forEach(el => {{
  el.style.setProperty('--tok-color', el.dataset.colorTok);
}});
document.querySelectorAll('.sent').forEach(el => {{
  el.style.setProperty('--sent-color', el.dataset.colorSent);
}});
document.querySelectorAll('.para').forEach(el => {{
  el.style.setProperty('--para-color', el.dataset.colorSent);
}});

const container = document.getElementById('content');
const tabs = document.querySelectorAll('.tab');
const sliderTok = document.getElementById('slider-token');
const sliderSent = document.getElementById('slider-sentence');
const legendTok = document.getElementById('legend-token');
const legendSent = document.getElementById('legend-sentence');

tabs.forEach(tab => {{
  tab.addEventListener('click', () => {{
    const mode = tab.dataset.mode;
    tabs.forEach(t => t.classList.toggle('active', t === tab));
    container.className = 'mode-' + mode;
    sliderTok.classList.toggle('hide', mode !== 'token');
    sliderSent.classList.toggle('hide', mode !== 'sentence');
    legendTok.classList.toggle('hide', mode !== 'token');
    legendSent.classList.toggle('hide', mode !== 'sentence');
  }});
}});

// Token threshold slider
const tokSlider = document.getElementById('tok-threshold');
const tokLabel = document.getElementById('tok-threshold-val');
const toks = document.querySelectorAll('.tok');
tokSlider.addEventListener('input', () => {{
  const v = parseFloat(tokSlider.value);
  tokLabel.textContent = v.toFixed(1);
  toks.forEach(el => {{
    const isBreak = el.querySelector('br') !== null;
    el.classList.toggle('hidden', !isBreak && parseFloat(el.dataset.bits) < v);
  }});
}});

// Sentence threshold slider
const sentSlider = document.getElementById('sent-threshold');
const sentLabel = document.getElementById('sent-threshold-val');
const sents = document.querySelectorAll('.sent');
sentSlider.addEventListener('input', () => {{
  const v = parseFloat(sentSlider.value);
  sentLabel.textContent = v.toFixed(1);
  sents.forEach(el => {{
    el.classList.toggle('hidden', parseFloat(el.dataset.bits) < v);
  }});
}});
</script></body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cliché Detector — measure token-level entropy of text using a language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python cliche_detector.py input.txt
  python cliche_detector.py input.txt -o report.html
  python cliche_detector.py input.txt --model Qwen/Qwen3.5-0.8B-Base --top-k 20
""",
    )
    parser.add_argument("input", help="Input plaintext file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML file (default: <input>.html)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base",
                        help="HuggingFace model ID (default: Qwen/Qwen3.5-0.8B-Base)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k alternatives to store (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Tokens per forward pass, lower = less VRAM (default: 4096)")
    args = parser.parse_args()

    input_file = args.input
    html_file = args.output or os.path.splitext(input_file)[0] + ".html"
    out_dir = os.path.dirname(html_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    text_bytes = len(text.encode("utf-8"))
    tokens_data = analyze(text, args.model, args.top_k, chunk_size=args.chunk_size)
    num_tokens = len(tokens_data)

    bits_list = [-t["logp"] / math.log(2) for t in tokens_data]
    total_bits = sum(bits_list)
    avg_bits = total_bits / num_tokens
    ratio = text_bytes / (total_bits / 8)

    compressed_bytes = total_bits / 8
    stats_text = (
        f"{args.model} | {num_tokens} tokens | "
        f"{text_bytes} bytes → {compressed_bytes:.0f} bytes "
        f"({avg_bits:.2f} bit/tok, {ratio:.1f}x)"
    )
    print(stats_text)

    paragraphs = split_paragraphs(tokens_data, bits_list)
    base = os.path.splitext(os.path.basename(input_file))[0]
    html_content = generate_html(tokens_data, bits_list, paragraphs, stats_text, title=base)
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Wrote {html_file}")


if __name__ == "__main__":
    main()
