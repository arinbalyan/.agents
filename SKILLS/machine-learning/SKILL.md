# Machine Learning Workflow Skill

Single skill covering the full ML lifecycle: Kaggle → train → HF Spaces.

**Always use `uv` — never pip directly.** This applies to kaggle, hf, colab CLI, and any Python dependency.

---

## Authentication Setup

### Kaggle
```bash
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
echo "KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```
Verify: `uv tool run kaggle competitions list`

### Hugging Face
```bash
hf auth login   # uses HF_TOKEN from ~/.cache/huggingface/token
# OR: export HF_TOKEN="hf_..."
```
Verify: `hf auth whoami`

### Google Colab CLI
```bash
uv tool install google-colab-cli
colab auth login   # browser-based OAuth
```

---

## Tool Installation (uv only)

| Tool | Install | Run |
|------|---------|-----|
| **kaggle** | `uv add kaggle kagglehub` (project) | `uv run kaggle ...` or `uv tool run kaggle ...` |
| **hf** | `curl -LsSf https://hf.co/cli/install.sh \| bash` | `hf ...` (system) or `uv run hf ...` (project) |
| **colab** | `uv tool install google-colab-cli` | `colab ...` |

---

## Core Workflows

### Kaggle: Train on GPU
1. Push notebook: `kaggle kernels push` (from dir with `kernel-metadata.json`)
2. Check status: `kaggle kernels status <user>/<kernel>`
3. Submit: `kaggle competitions submit -c <comp> -f submission.csv -m "msg"`
4. Download output: `kaggle kernels output <user>/<kernel> -p /tmp/out`

### HF: Host Model + Space
1. Create model repo: `hf repos create <name> --type model --public`
2. Upload: `hf upload <user>/<repo> <file> --type model`
3. Deploy space: `hf repos create <name> --type space --space-sdk gradio`
4. Upload app: `cd space/ && hf upload <user>/<space> app.py --type space`
5. Restart: `hf spaces restart <user>/<space>`
6. Check logs: `hf spaces logs <user>/<space>`

### Colab: Quick GPU Runtime
```bash
colab new -s my-session                    # create session
colab list                                 # list running sessions
colab upload my-session notebook.ipynb      # upload notebook
colab run my-session                       # execute
colab download my-session output.csv       # get results
```

---

## P100 GPU Install (Kaggle Notebooks)

The ONLY working install for P100 (Pascal sm_60):
```python
!pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
!pip uninstall -y torchao 2>/dev/null
!pip install -q --upgrade "transformers>=4.45,<4.50" datasets accelerate \
  "peft>=0.12,<0.15" trl scikit-learn scipy pandas
```

**Rules:** No `--no-deps`, no numpy pinning, no torchao, no bitsandbytes.

---

## Directory Conventions

```
machine-learning/
├── projects/NN-name/      # Portfolio projects
│   ├── space/             # HF Space (app.py, requirements.txt)
│   ├── kaggle/            # Training notebook + kernel-metadata.json
│   ├── README.md
│   └── LEARNING.md
├── kaggle-competitions/   # Active comps
│   └── <name>/README.md
└── scripts/               # Pipeline helpers
```

See `references/` for: full Kaggle CLI reference, HF Spaces deployment gotchas, Colab CLI patterns, dependency debug history, P100 fine-tuning templates, notebook JSON pitfalls, and competition submission playbook.
