# Machine Learning Workflow Skill

Single skill covering the full ML lifecycle: Kaggle → training → HF Spaces.
All tools run via `uv` — never pip directly.

---

## Authentication (one-time setup)

```bash
# Kaggle
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
echo "KGAT_xxx" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# Hugging Face
hf auth login    # or: export HF_TOKEN="hf_..."

# Colab (optional — for quick GPU)
uv tool install google-colab-cli && colab auth login
```

Verify: `uv tool run kaggle competitions list` | `hf auth whoami`

---

## Tool Usage (uv only)

| Action | Command |
|--------|---------|
| **Kaggle quick status** | `uv tool run kaggle competitions list` |
| **Kaggle project push** | `uv run kaggle kernels push` (from metadata dir) |
| **Kaggle kernel status** | `uv run kaggle kernels status <user>/<kernel>` |
| **Kaggle submit** | `uv run kaggle competitions submit -c <comp> -f <file> -m "msg"` |
| **Kaggle submissions** | `uv run kaggle competitions submissions <comp>` |
| **Kaggle leaderboard** | `uv run kaggle competitions leaderboard -c <comp> --show` |
| **Kaggle datasets** | `uv run kaggle datasets download <user>/<ds> -p /tmp` |
| **Kaggle kernel output** | `uv run kaggle kernels output <user>/<kernel> -p /tmp/out` |
| **HF upload** | `hf upload <user>/<repo> <file> --type model\|space` |
| **HF space info/logs** | `hf spaces info\|logs <user>/<space>` |
| **HF space restart** | `hf spaces restart <user>/<space>; hf spaces wait <user>/<space>` |
| **Colab run** | `colab new -s name && colab upload name nb.ipynb && colab run name` |

**Gotcha — uv vs uv tool:** Inside a project dir with active venv, `uv run kaggle` uses the project's deps. `uv tool run kaggle` uses a global tool cache — preferred for quick one-off commands after `uv cache clean`.

---

## P100 Kaggle Notebook Install (THE ONLY WORKING SEQUENCE)

Verified across 90+ notebook versions. Do NOT deviate from this:

```python
# Step 1: torch from CUDA 11.8 (Pascal sm_60 compatible)
!pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118

# Step 2: torchao breaks everything on P100 — needs torch 2.6+
!pip uninstall -y torchao 2>/dev/null

# Step 3: --upgrade fixes numpy ABI mismatch
!pip install -q --upgrade "transformers>=4.45,<4.50" datasets accelerate \
  "peft>=0.12,<0.15" trl scikit-learn scipy pandas
```

**Rules:** ✗ `--no-deps` → breaks Trainer import. ✗ numpy pins → breaks scikit C extensions. ✗ torchao → needs torch 2.6+. ✗ bitsandbytes → no sm_60 support. ✗ `--force-reinstall` → triggers numpy downgrade cascade.

For CPU/sklearn notebooks: use zero-dep approach (Kaggle base env has everything). No pip install cell needed.

---

## Learnings (hard-won over the ML project lifecycle)

### Research-First Debug Rule
When a Kaggle kernel errors, do NOT push 20+ versions guessing. Read the full error log → web search the exact error → fix ALL broken deps in ONE push. After 2 failed pushes, stop and research the full dependency chain end-to-end.

### Notebook JSON Pitfalls
- Every `id` must be a non-null string (use `uuid.uuid4().hex[:8]`)
- `source` must be a **list of strings** (each with trailing `\n`), NOT a single string
- `kernelspec` metadata must exist: `{"display_name": "Python 3", "language": "python", "name": "python3"}`
- `outputs` in markdown cells breaks re-push — strip them with `del cell['outputs']`
- Use `json.dump(nb, f, indent=1)` to write — NOT patch/sed (corrupts `\n` escaping)
- Validate pre-push: `compile()` on every code cell's source

### Competition Data Hard Rule
NEVER download competition data to the local VM (29GB disk). All data processing happens INSIDE the Kaggle notebook. VM only pushes, submits, and checks status.

### Submission Strategy
For tabular/synthetic competitions, top public submissions converge to ~99.9% identical predictions. Ensemble methods can't improve — the best path is finding better individual submissions. For ONNX-based comps, validate against actual task examples, not op-type filters.

### HF Space Deployment
- Upload from a fresh temp dir (avoids dedup hash-matching bugs)
- Pin `transformers>=5.0` not `<5.0` — gradio build system needs `huggingface-hub>=1.2`
- Always include `torchvision` for VLM Spaces
- Gradio 6.0: pass `theme` + `css` to `launch()`, not `Blocks()` constructor
- Monkey-patch `gradio_client.utils.get_type` for 5.9.0 bug

### Polling Pattern
Monitor kernels at 30s+ intervals. Use `background=true notify_on_complete=true` for single-notebook monitoring. For batch monitoring, Python polling loop with status dict.

### Kaggle Metadata Booleans
Use **string** booleans in `kernel-metadata.json`: `"true"` / `"false"` — not JSON `true`/`false`. The kaggle CLI template uses strings.

### Slug Gotcha
Kernel slug is derived from `title`, not `id`. After first push, `kaggle kernels pull` to get actual metadata.

### GPU Quota
2 concurrent GPU sessions max, 30h/week. When full: push with `enable_gpu: false`, then enable in UI.
