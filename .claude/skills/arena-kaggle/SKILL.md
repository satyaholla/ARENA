---
name: arena-kaggle
description: Convert or repair ARENA exercise notebooks so they run on Kaggle GPU notebooks, including Kaggle Secrets, dependencies, paths, caches, accelerator constraints, and validation. Use for requests to make ARENA notebooks Kaggle-friendly or diagnose Colab-specific failures on Kaggle.
---

# ARENA → Kaggle notebook conversion

## Scope and source of truth

- Target the notebook the user named in `satyaholla/ARENA`. Modify the existing `.ipynb` **in place**, not a separate Kaggle copy, unless the user explicitly asks for a copy.
- Preserve exercise text, problem statements, function signatures, user solutions, tests, notebook cell IDs/order, and pedagogical logic. Change **only** setup/infrastructure and necessary compatibility code. Keep a minimal, inspectable diff.
- Read the notebook's actual setup and all later cells that read API keys or access Colab services before editing. Avoid assuming that fixing the initial setup cell automatically fixes all later accesses.
- GitHub is the source of truth for the user's ARENA notebooks. Kaggle notebook edits / Quick Save are not automatically synchronized to GitHub; any edits made on Kaggle should be exported back to the corresponding GitHub notebook.

## Mandatory Kaggle Secrets procedure

1. Identify **every** secret used by the notebook (`HF_TOKEN`, `WANDB_API_KEY`, etc.) by searching for `os.getenv`, `os.environ`, `userdata.get`, `get_secret`, client login calls, and API-key parameters.
2. Give the user explicit Kaggle instructions: open **Add-ons → Secrets**, create a secret with the exact required name (e.g. `HF_TOKEN`), paste its value, and **enable/attach the secret for this notebook**. Enable notebook **Internet** when downloads/API access are needed. For gated Hugging Face weights, the HF account owning the token must separately have accepted the model's terms/access request.
3. On Kaggle, retrieve the secret using `kaggle_secrets.UserSecretsClient().get_secret(name)`. Export it to `os.environ[name]` so the Hugging Face Hub and other libraries that use environment variables see it. Never print/log the value, paste the token into notebook source, or commit a `.env` containing credentials.
4. **At every gated-model/API loading point**, ensure the key has been retrieved and present: a Colab setup helper or `load_dotenv()` alone is *not* sufficient on Kaggle. Provide a helpful failure message explaining how to grant notebook secret access.
5. Preserve non-Kaggle fallback through an existing environment variable or local `.env`. Do not import `kaggle_secrets` unconditionally in a local/Colab run.

Suggested idempotent helper, installed near the notebook bootstrap:

~~~python
import os
from pathlib import Path

IN_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").is_dir()

def load_runtime_secret(name: str, *, required: bool = False) -> str | None:
    value = os.environ.get(name)
    if not value and IN_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            value = UserSecretsClient().get_secret(name)
        except Exception as exc:
            if required:
                raise RuntimeError(
                    f"Cannot access Kaggle secret {name}. Create it under "
                    "Add-ons > Secrets and enable access for this notebook."
                ) from exc
    if value:
        os.environ[name] = value
    if required and not value:
        raise RuntimeError(
            f"Missing {name}. On Kaggle, set it under Add-ons > Secrets "
            "and enable notebook access; elsewhere, use an environment variable or .env."
        )
    return value
~~~

At the point of use (e.g. before `HookedSAETransformer.from_pretrained` with gated Gemma weights):

~~~python
if not IN_KAGGLE:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(exercises_dir / ".env"))
HF_TOKEN = load_runtime_secret("HF_TOKEN", required=True)
# Never print HF_TOKEN. Libraries can now read os.environ["HF_TOKEN"].
~~~

When a notebook needs a different key, call the same helper with its exact secret name. Do **not** assume one secret covers every API.

## Kaggle bootstrap and dependencies

- Ensure `arena_lib`, ARENA exercise modules, test modules, and `exercises_dir` / `section_dir` resolve in a fresh Kaggle notebook (not only in Colab or a local clone). Use the repository's actual structure and stable paths; do not hardcode `/content`.
- Kaggle may need manual dependency installation even when `arena_lib.setup()` handles it on Colab. Prefer the repo's pinned `requirements.txt` and `arena_lib/day_packages.generated.json` for that exact exercise. Verify the selected packages and compatibility, avoid unnecessary global upgrades, and state when a kernel restart is required.
- Replace Colab-only `google.colab` imports, `drive.mount`, `userdata.get`, `files.download`, and Colab-specific HTML serving **only along the Kaggle execution path**. Use Kaggle-supported input/output paths (`/kaggle/input` for attached read-only inputs, `/kaggle/working` for writable outputs) and notebook-native displays or explicit file links.
- Enable Kaggle Internet for `git`/pip/Hugging Face downloads; check available disk space before large checkpoints. A Kaggle kernel's disk budget and HF cache locations can be more constraining than the compute.
- Keep local/Colab behavior functioning when reasonably possible via runtime detection.

## GPU and RAM guardrails

- Kaggle T4/P100: roughly 16 GiB **per GPU**; dual T4s are not one 32 GiB device. Read actual available memory with `torch.cuda.mem_get_info()`. Kaggle host RAM and disk also have independent limits.
- Provide a `LOW_GPU_MEM` path where model size requires it. Start by reducing batch sizes and sequence lengths, processing Jacobians/attributions in chunks, computing only the required output slices, and freeing unused models/tensors with `del` and `gc.collect()`.
- `torch.cuda.empty_cache()` only clears unused cached allocations; it does not free live tensors.
- T4/P100 do not support native bf16. Use fp16 only for components verified to tolerate it, and document possible numerical changes; do not blindly change the exercise's dtype or test expectations. Preserve float32 when feasible.
- Avoid materializing full dense `d_sae × d_sae` matrices or differentiating dense auxiliary outputs. For `torch.func.jacrev`, check `has_aux=True` where intended; use batches/chunks and sparse active-latent indexing.

## Validation before declaring success

1. Confirm the modified notebook is parseable notebook JSON, with all cells and metadata intact; do not unintentionally strip outputs/solutions.
2. Inspect the diff: preserve exercise logic and keep infrastructure changes localized.
3. Confirm no remaining **unconditional Colab-only** import in the Kaggle path; check every secret read, **including later model loading cells**, to ensure it uses `load_runtime_secret` or equivalent.
4. Confirm setup resolves local exercise imports, dependency versions and test files; check memory settings, model download/gating instructions, and output paths.
5. Run static checks and small non-secret smoke tests where tools allow. If no Kaggle GPU/secrets are available, explicitly say it was **not executed on Kaggle**; never claim an end-to-end successful Kaggle run.
6. Commit the edited original notebook to the user's GitHub repo and report which paths changed. Do not include secrets in commits.

## ARENA 1.4.2 special case

- Notebook: `chapter1_transformer_interp/exercises/part42_sae_circuits/1.4.2_SAE_Circuits_exercises.ipynb`.
- Section 3 downloads gated `google/gemma-3-1b-it` and GemmaScope 2 transcoders. **Load `HF_TOKEN` directly from Kaggle Secrets at its model-loading cell**, not only in setup and not solely via `.env`.
- Section 3 uses all 26 16k transcoders, so use low-memory backward batches on T4 (e.g. 2, then 1 if necessary); do not silently remove layers or alter learning exercises.
- Sections 3/4 require substantial downloads and their combined model caches may exceed disk space. Section 4's Gemma 2-2B setup may need fp16 on T4/P100, with a numerical-tolerance warning.
