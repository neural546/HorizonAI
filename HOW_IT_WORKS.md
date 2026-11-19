
**Job state machine:** `queued → running → completed | failed`.

**Model caching.** `_get_or_build_model` keys the cache on `(device, image_size, num_frames, vocab_size, tokenizer_fingerprint)`. The cache is an `OrderedDict` capped at 3 entries (`MAX_MODEL_CACHE`); the least-recently-used model is evicted.

**Test short-circuit.** When `app.config["TESTING"]` is set, `_run_generation` immediately marks the job completed without running a real model. Unit tests still exercise the full API contract; end-to-end generation is covered by smoke tests directly.

---

## 6. Module Reference

### `AI/tokenizer.py`

**Purpose.** Convert text into fixed-length integer token sequences and back.

**Data structure.** `token_to_id: dict[str, int]`, `id_to_token: dict[int, str]`. The first four ids are always the special tokens `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.

| Function | What it does | When it's called |
|----------|--------------|------------------|
| `_tokenize_text(text)` | Splits on word boundaries and punctuation, lowercases. | Internally by `encode` and `build_vocab`. |
| `SimpleTokenizer.__init__(vocab, max_length)` | Builds the id↔token maps. | Whenever a tokenizer is constructed or loaded. |
| `build_vocab(texts, min_freq=1)` | Counts tokens across `texts`, adds them alphabetically to the vocab. | First training run only. Subsequent runs load from `tokenizer.json`. |
| `save(path)` / `load(path)` | JSON round-trip of `{max_length, vocab}`. | After `ensure_tokenizer` creates a new vocab; on every inference load. |
| `encode(text, add_special_tokens=True)` | Returns `[BOS, tokens…, EOS]` truncated to `max_length`, padded with `pad_id`. | Every prompt the model sees. |
| `decode(ids, skip_special_tokens=True)` | Inverse. Rarely used outside debugging. | Rarely. |
| `attention_mask(ids)` | `1` for real tokens, `0` for pad. | Passed alongside `encode` output to the text encoder. |
| `build_default_tokenizer(max_length)` | Returns a tokenizer pre-populated with ~19 sample prompts. | Fallback when no `tokenizer.json` exists yet. |

**Invariant.** `len(encode(t)) == max_length` for any `t`.

---

### `AI/scheduler.py`

**Purpose.** Define the forward noising process (training) and the reverse denoising process (inference).

**Data structures.** Precomputed at construction:
- `_betas`, `_alphas`, `_alphas_cumprod`, `_alphas_cumprod_prev`
- `_sqrt_alphas_cumprod`, `_sqrt_one_minus_alphas_cumprod`
- `_posterior_mean_coef1`, `_posterior_mean_coef2`, `_posterior_variance`

All are `torch.float32` tensors on `self.device`.

| Function | What it does | When it's called |
|----------|--------------|------------------|
| `_make_beta_schedule(schedule, T, β_start, β_end)` | Linear or cosine β schedule. | Once, at construction. |
| `DDPMScheduler.__init__(...)` | Precomputes every coefficient. | Once per training/inference run. |
| `_extract(arr, t, shape)` | Gathers `arr[t[b]]` per batch item and reshapes to broadcast. | Inside `add_noise` and `step`. |
| `add_noise(x₀, ε, t)` | Returns `x_t = √α̅·x₀ + √(1−α̅)·ε`. | Training only. |
| `sample_timesteps(B, device)` | Random `t ∈ [0, T)`, shape `[B]`. | Training only. |
| `_make_timestep_list(n)` | Evenly-spaced descending timesteps. `n = 1` returns `[T-1]`. | Inference only. |
| `DDPMScheduler.step(ε̂, t, xₜ, gen)` | One reverse DDPM step. Adds noise (stochastic). | Only via `DDPMScheduler.sample`. |
| `DDPMScheduler.sample(fn, shape, n)` | Full DDPM loop. | Rarely used — training uses `add_noise`, not `sample`. |
| `DDIMScheduler.step(ε̂, t, xₜ, prev_t, gen)` | Deterministic (η=0) or stochastic (η>0) reverse step. | Every inference step. |
| `DDIMScheduler.sample(fn, shape, n)` | Full DDIM loop. | Convenience for testing. |

**Invariant.** `add_noise(x₀, ε, 0) ≈ x₀` (because α̅₀ ≈ 1). Verified by `test_add_noise_at_t0_close_to_original`.

**Numerical trick.** `_extract` returns `float32`; callers cast to `sample.dtype` right before multiplication. This keeps AMP-safe.

---

### `AI/model.py`

**Purpose.** Everything that has learnable weights lives here.

**Sub-modules and what they do:**

| Class | Role | Input → Output |
|-------|------|----------------|
| `TextTransformerEncoder` | Token + positional embeddings, then N transformer layers. | `[B, L] int` + mask → `[B, L, text_embed_dim]` |
| `TimeEmbed` | Sine embedding + 2-layer MLP. | `[B] t` → `[B, time_dim·4]` |
| `CrossAttention` | Standard multi-head cross-attention. Q from video, K/V from text. | `[B,N,dim] × [B,L,ctx_dim]` → `[B,N,dim]` |
| `ResBlock3D` | GroupNorm+SiLU+Conv3D, injects time embedding, residual skip. | `[B,C,T,H,W] × [B,td]` → `[B,C',T,H,W]` |
| `TemporalAttention` | Self-attention across the `T` axis per `(h, w)` location. | `[B,C,T,H,W]` → same |
| `SpatialCrossAttention` | Self-attention across `H·W` per frame + cross-attention to text. | `[B,C,T,H,W] × [B,L,ctx]` → same |
| `VideoUNet` | Assembles all of the above into a U-Net with one skip per level. | `[B,4,T,h,w] × [B] × [B,L,ctx]` → `[B,4,T,h,w]` |
| `TinyVAEEncoder` | `log2(downsample)` stride-2 Conv3D layers. | `[B,3,T,H,W]` → `[B,4,T,H/8,W/8]` |
| `TinyVAEDecoder` | Mirror with nearest-neighbor upsampling and `tanh`. | `[B,4,T,H/8,W/8]` → `[B,3,T,H,W]` in `[-1,1]` |
| `TextToVideoModel` | Wraps text encoder + U-Net + VAE. | Full pipeline |

**Key functions:**

| Function | What it does |
|----------|--------------|
| `timestep_embedding(t, dim)` | Sinusoidal positional encoding, `[B] → [B, dim]`. |
| `VideoUNet.forward(x, t, context)` | Full forward pass. Pushes one skip per encoder level, pops one per decoder level, interpolates if shapes disagree. |
| `VideoUNet._res(rb, h, t_emb)` | Runs a `ResBlock3D`, optionally wrapped in `checkpoint.checkpoint` when `self.gradient_checkpointing and self.training`. |
| `VideoUNet._project_context(ctx)` | Pads or truncates text embeddings to `self.context_dim`. Defensive — normally a no-op. |
| `TextToVideoModel.encode_video(frames)` | Permute to `[B,C,T,H,W]` and run the VAE encoder. |
| `TextToVideoModel.decode_video(latents)` | Run the VAE decoder and permute back to `[B,T,3,H,W]`. |
| `TextToVideoModel.forward(noisy, t, ids, mask)` | `encode_text` + `unet` — this is the training loss input. |
| `TextToVideoModel.set_gradient_checkpointing(bool)` | Toggle checkpointing without mutating every submodule individually. |

**Invariant.** `TextToVideoModel.decode_video(model.encode_video(frames)).shape == frames.shape` — verified in `test_encode_decode_video`.

---

### `AI/dataset.py`

**Purpose.** Read and validate the on-disk dataset, and expose it as a PyTorch `Dataset`.

**On-disk layout.** Each sample is a directory containing one `prompt.txt` plus `NNNNNN.{png,jpg,jpeg}` files named with **sequential zero-padded integers starting at 1**.

**Report dataclasses.**

| Dataclass | Fields |
|-----------|--------|
| `SampleReport` | `sample_id`, `valid`, `errors: List[str]`, `num_frames`, `resolution: (w,h)` |
| `ValidationReport` | `root`, `total_samples`, `valid_samples`, `invalid_samples`, `samples` |

**Validation functions:**

| Function | What it does |
|----------|--------------|
| `_discover_frames(dir)` | Enumerates `NNNNNN.ext` files, detects duplicates, gaps, and out-of-order numbering. Returns `(paths, errors)`. |
| `_read_image_rgb(path)` | Opens and converts to RGB. Returns `None` on corruption. |
| `validate_sample(dir)` | Runs all checks: prompt present/non-empty, ≥ 8 and ≤ 1024 frames, all readable, all same resolution, no duplicates (SHA-1 of raw bytes), numbering sequential. |
| `validate_dataset(root)` | Iterates subdirectories and calls `validate_sample`. Returns aggregate `ValidationReport`. |
| `repair_dataset(root, remove_invalid=True)` | Removes unsupported/corrupt files, renumbers remaining frames, drops samples with no prompt or no frames. Returns a dict of `removed_files`, `renamed`, `removed_samples`. |

**Dataset class:**

| Method | What it does |
|--------|--------------|
| `VideoTextDataset.__init__(root, image_size, num_frames, tokenizer, max_length, validate)` | Enumerates subdirectories with a `prompt.txt`. |
| `__len__()` | Number of samples. |
| `_load_frames(dir)` | Discovers frames, uniformly samples `num_frames` of them, resizes to `image_size × image_size`. |
| `__getitem__(i)` | Returns `{"frames": [T,3,H,W] float in [-1,1], "prompt": str, "input_ids": [L] long, "attention_mask": [L] long}`. |

**Invariant.** Every batch item has exactly `num_frames` frames even if the source sample had 30.

---

### `AI/collate.py`

**Purpose.** Turn a list of dataset items into a single batched dict.

| Function | What it does |
|----------|--------------|
| `collate_fn(batch)` | `torch.stack` each tensor key. `"prompt"` stays as `List[str]`. Raises `ValueError` on empty batch. |
| `random_prompt_dropout(input_ids, mask, pad_id, prob, gen)` | Returns `[B]` float tensor; `1.0` means "drop this prompt". `prob = 0` always returns zeros. |

**Invariant.** The output of `collate_fn` has `frames` shape `[B, T, 3, H, W]`.

---

### `AI/train.py`

**Purpose.** The training entry point and loop.

| Function | What it does |
|----------|--------------|
| `set_global_seed(seed)` | Seeds `random`, `numpy`, `torch`, `torch.cuda`, sets `PYTHONHASHSEED`, forces cuDNN deterministic. |
| `load_yaml(path)` | Simple `yaml.safe_load`. |
| `resolve_profile(model_cfg, train_cfg, profile, auto)` | If `auto`, calls `resource_detector.detect_profile()`. Returns `(merged_model_cfg, merged_train_cfg, profile_name)`. Merging is `defaults ∪ profiles[name]`, with profile entries winning. |
| `ensure_tokenizer(dataset_root, checkpoint_dir, max_length)` | Loads `tokenizer.json` if present, else builds one from the dataset prompts and saves. **Must match what inference loads.** |
| `_save_checkpoint(path, model, optimizer, epoch, step)` | Writes a dict `{"model", "optimizer", "epoch", "global_step"}`. |
| `train(args, epochs_override=None)` | The full loop (see [§3](#3-training-flow)). |
| `_main(argv)` | Parses CLI flags, forwards to `train`. |

**CLI flags:** `--profile {low,medium,high}`, `--auto-profile`, `--seed`, `--data-dir`, `--checkpoint-dir`, `--resume`, `--epochs`.

**Checkpoints written:** `AI/checkpoints/latest.pt` at every `save_every` steps and at the end; plus `step_XXXXXX.pt` snapshots.

---

### `AI/inference.py`

**Purpose.** Generate a video from a prompt.

| Function | What it does |
|----------|--------------|
| `set_global_seed(seed)` | Same as training. |
| `load_yaml`, `resolve_profile` | Same as training. |
| `load_tokenizer(checkpoint_dir, max_length)` | Loads `tokenizer.json`, falls back to `build_default_tokenizer`. |
| `generate_video(prompt, model, tokenizer, scheduler, image_size, num_frames, num_inference_steps, cfg_scale, device, seed, progress_callback)` | The CFG sampling loop (see [§4](#4-inference-flow)). Returns `[1, T, 3, H, W]` in `[-1, 1]`. |
| `_main(argv)` | CLI: loads configs, builds model, loads checkpoint, calls `generate_video`, calls `export_video`. |

**CLI flags:** `--prompt`, `--seed`, `--profile`, `--auto-profile`, `--cfg-scale`, `--num-frames`, `--num-inference-steps`, `--image-size`, `--fps`, `--output`, `--checkpoint`, `--device`.

**Invariant.** `progress_callback(i, total)` is called once per timestep, exactly `total` times.

---

### `AI/export.py`

**Purpose.** Turn a tensor (or list) of frames into an H.264 MP4 using FFmpeg.

| Function | What it does |
|----------|--------------|
| `_to_uint8_frames(frames)` | Accepts `[T,C,H,W]`, `[B,T,C,H,W]`, or `List[H,W,3]`. Normalizes `[-1,1]` → `[0,255]`, transposes to `H,W,3`, returns `List[np.uint8]`. |
| `_ensure_ffmpeg()` | Checks `shutil.which("ffmpeg")`, raises a helpful `RuntimeError` if missing. |
| `export_video(frames, output_path, fps, bitrate, resolution)` | Writes PNGs to a temp dir, runs `ffmpeg -framerate fps -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p -b:v bitrate -movflags +faststart`. Returns absolute path. |
| `frames_from_video(path)` | Inverse — decodes a video to RGB frames. Used in tests. |

**Invariant.** `export_video` guarantees even width/height (truncates by 1 pixel if needed) — `libx264` with `yuv420p` requires it.

---

### `AI/resource_detector.py`

**Purpose.** Inspect hardware and pick a `low`/`medium`/`high` profile.

| Function | What it does |
|----------|--------------|
| `_get_ram_gb()` | Tries `psutil`, then `/proc/meminfo`, falls back to `8.0`. |
| `detect_resources()` | Returns `ResourceInfo(cpu_count, ram_gb, cuda_available, vram_gb)`. |
| `detect_profile(info)` | Rules: CUDA + ≥ 15 GB VRAM → `high`; CUDA + ≥ 7 GB → `medium`; CUDA → `low`; CPU + ≥ 32 GB RAM + ≥ 8 cores → `medium`; otherwise `low`. |
| `report()` | Human-readable string for `--report`. |

---

### `AI/lightweight/`

**Purpose.** A faster, smaller variant that fits on CPU or a 4 GB GPU.

| Module | Function | Notes |
|--------|----------|-------|
| `tokenizer.py` | `build_mvp_tokenizer(max_length=16)` | Same vocab as the default tokenizer, shorter max length. |
| `model.py` | `LightVideoUNet` | One ResBlock per level (instead of `num_res_blocks`), half the channels. |
| `model.py` | `LightTextToVideoModel` | `base_channels = 32`, `text_embed_dim = 128`, VAE `base_ch = 16`. |
| `train.py` | `train_mvp(steps, seed)` | Trains on **synthetic** `[1, 8, 3, 64, 64]` random frames — no dataset needed. Saves `lightweight_latest.pt`. |
| `inference.py` | `load_mvp_model(checkpoint, device, num_frames)` | Loads `tokenizer.json` first (so vocab size matches), then builds the model. |
| `inference.py` | `generate_mvp(prompt, cfg_scale, num_inference_steps, seed, output, device, num_frames, image_size, fps)` | Same CFG sampling as the full model, hardcoded to smaller defaults. |

**Why the MVP exists.** The full model with `num_res_blocks=2` on CPU takes minutes per step. The MVP completes in seconds. Use it for demos, CI smoke tests, and any environment without a GPU.

**Note.** `train_mvp` writes `tokenizer.json` **and** `lightweight_latest.pt`. `app.py` loads `latest.pt` (the full model), not the MVP — so the web UI **requires** either a full checkpoint or a retrain via `AI/train.py`.

---

### `app.py`

**Purpose.** Flask HTTP API and background job orchestration.

**State.**

| Global | Type | Purpose |
|--------|------|---------|
| `JOBS` | `Dict[str, Dict]` | Job id → job record. |
| `JOB_LOCK` | `threading.Lock` | Serializes job mutations. |
| `MODEL_CACHE` | `OrderedDict` | LRU of loaded models. |
| `MODEL_CACHE_LOCK` | `threading.Lock` | Serializes cache access. |
| `MAX_MODEL_CACHE` | `int = 3` | Cache size cap. |

**Helpers:**

| Function | What it does |
|----------|--------------|
| `_prune_jobs()` | Removes jobs older than `JOB_TTL_SECONDS` (1 h). Called on every new job. |
| `_new_job(prompt, cfg, frames, size, fps, seed)` | Allocates a uuid, stores the record. Returns `job_id`. |
| `_update_job(id, **kwargs)` | Thread-safe merge. |
| `_get_job(id)` | Returns a copy of the job record or `None`. |
| `_tokenizer_fingerprint(tok)` | SHA-1 of the vocab mapping — prevents a cache hit against a differently-ordered tokenizer. |
| `_get_or_build_model(device, image_size, num_frames, tok, m_cfg, model_cfg)` | Builds a `TextToVideoModel`, loads `latest.pt` if present, stores in LRU. |
| `_run_generation(job_id)` | Runs in a worker thread. Test-short-circuits when `app.config["TESTING"]`. Otherwise loads, samples, exports, updates job. |

**Routes:**

| Route | Handler | Returns |
|-------|---------|---------|
| `GET /` | `index` | `Frontend/index.html` |
| `GET /health` | `health` | `{"status": "ok"}` |
| `POST /generate` | `generate` | `202 {job_id, status: "queued"}` or `400 {error}` |
| `GET /progress?job_id=…` | `progress` | `{status, progress, message, video_url, error}` |
| `GET /video` | `video_latest` | Latest `.mp4` in `static/videos/` |
| `GET /video/<file>` | `video_file` | Serves the named file from `static/videos/` |

**Validation rules in `/generate`:**

- `prompt` non-empty string → else 400
- `cfg_scale` ∈ `[1.0, 20.0]`
- `num_frames` ∈ `[8, 1024]`
- `image_size` ∈ `{64, 128, 256}`
- `fps` ∈ `[1, 60]`
- `seed` any int (default 42)

---

### `health_check.py`

**Purpose.** Pre-flight diagnostics. Exit `0` = passed, `1` = failed.

| Function | What it checks |
|----------|----------------|
| `check(label, fn)` | Runs `fn() → (ok, detail)`; records failure as error. |
| `warn(label, fn)` | Same but records as warning (non-fatal). |
| `check_imports()` | Imports `torch`, `numpy`, `PIL`, `yaml`, `flask`, `imageio`, `diffusers`, `transformers`, `accelerate`. |
| `check_configs()` | `config/*.yaml` present. |
| `check_tokenizer()` | `build_default_tokenizer(16)` produces sane output. |
| `check_dataset_module()` | `validate_dataset()` runs on `tests/data/`. |
| `check_model_import()` | `AI.model` imports cleanly. |
| `check_scheduler()` | DDPM + DDIM construct. |
| `check_ffmpeg()` | `ffmpeg` on `PATH`. |
| `check_frontend()` | `Frontend/index.html`, `static/style.css`, `static/app.js` present. |
| `warn("CUDA", check_cuda())` | Warns if `torch.cuda.is_available()` is `False`. |
| `warn("Checkpoints", check_checkpoints())` | Warns if `AI/checkpoints/latest.pt` is missing. |
| `main()` | Runs all checks, prints `HEALTH CHECK PASSED` or `HEALTH CHECK FAILED` with reasons. |

---

## 7. Config Reference

### `config/model.yaml`

| Key | Where used | Meaning |
|-----|------------|---------|
| `image_size` | `TextToVideoModel`, `VideoTextDataset` | Spatial size of source frames. |
| `num_frames` | same | Frames per sample. |
| `latent_channels` | model | VAE latent channels (default 4). |
| `base_channels` | model | First-conv U-Net width. |
| `channel_multipliers` | model | Per-level U-Net width multipliers. |
| `num_res_blocks` | model | ResBlocks per encoder level. |
| `attention_heads` | model | Heads for temporal + spatial attention. |
| `text_max_length` | tokenizer | Sequence length fed to text encoder. |
| `text_embed_dim` | model | Width of text embeddings. |
| `transformer_layers` | model | Depth of text encoder. |
| `transformer_heads` | model | Heads in text encoder. |
| `dropout` | model | Dropout in text encoder and ResBlocks. |
| `vae.downsample_factor` | model | `log2` factor for spatial compression (8 → 64×64 frames become 8×8 latents). |

### `config/training.yaml`

| Key | Meaning |
|-----|---------|
| `epochs` | Number of passes over the dataset. |
| `batch_size` | Samples per step (profile-dependent). |
| `learning_rate`, `betas`, `weight_decay` | AdamW hyperparameters. |
| `grad_clip` | Max grad norm. |
| `log_every`, `save_every` | Frequency of TB logging and checkpoint writes. |
| `num_workers` | DataLoader workers. |
| `mixed_precision` | Enables AMP on CUDA. |
| `gradient_accumulation_steps` | Accumulate before optimizer.step. |
| `use_gradient_checkpointing` | Trades compute for memory in the U-Net. |
| `prompt_dropout_prob` | Probability a batch item's prompt is replaced with `""`. |
| `scheduler_type` | `linear` or `cosine` beta schedule. |
| `num_train_timesteps` | `T` in DDPM (default 1000). |

### `config/inference.yaml`

| Key | Meaning |
|-----|---------|
| `num_inference_steps` | Reverse-loop steps (profile-dependent). |
| `cfg_scale` | Default CFG scale. |
| `scheduler_type` | `ddim` or `ddpm`. |
| `eta` | DDIM stochasticity (`0` = deterministic). |
| `fps` | Output frame rate. |
| `bitrate` | FFmpeg `-b:v` value. |
| `output_path` | Default output MP4 path. |

---

## 8. Common Operations

### "Where do I change the prompt handling?"

`AI/tokenizer.py`. Everything else consumes whatever `encode()` returns.

### "Where do I add a new scheduler?"

Subclass `DDPMScheduler` in `AI/scheduler.py`, implement `step()`, and pass an instance to `generate_video`. Training is scheduler-agnostic as long as `add_noise` and `sample_timesteps` work.

### "How do I make the model bigger?"

Edit `config/model.yaml`, raise `base_channels`, `channel_multipliers`, `num_res_blocks`, `transformer_layers`. Everything else reads from config.

### "How do I add a new endpoint?"

Add a `@app.route(...)` in `app.py`. If the endpoint needs a model, use `_get_or_build_model` to reuse the cache. Long work must run in a thread (see `_run_generation`).

### "How do I reproduce an exact video?"

Same `--seed` + same checkpoint + same hardware + same torch version. Deterministic cuDNN is already forced by `set_global_seed`.

### "Why does `app.py` not use the MVP model?"

`_get_or_build_model` instantiates `TextToVideoModel` (the full model). To use the MVP in Flask, either:
1. Change `_get_or_build_model` to construct `LightTextToVideoModel`, or
2. Add an endpoint that calls `AI.lightweight.inference.generate_mvp`.

### "How does the progress bar work?"

`generate_video` accepts a `progress_callback(step, total)`. `app.py::_run_generation` passes a callback that translates `step/total` into `[0.05, 0.90]` of the job's `progress` field. The frontend polls `/progress` every 1.2 s.

### "How is reproducibility enforced?"

`set_global_seed(seed)` is called in `train.py`, `inference.py`, and `app.py::_run_generation`. It seeds every RNG and forces `torch.backends.cudnn.deterministic = True` and `benchmark = False`.

### "What does `_res()` do in the U-Net?"

It's the indirection layer for gradient checkpointing: if `self.gradient_checkpointing and self.training`, it wraps the call in `torch.utils.checkpoint.checkpoint(rb, h, t_emb, use_reentrant=False)`; otherwise it calls `rb(h, t_emb)` directly.

### "Why one skip per level instead of per res block?"

`VideoUNet` pushes the post-attention feature at each encoder level into `skips` and pops one per decoder level. This makes the up-path channel accounting **independent of `num_res_blocks`**, so `num_res_blocks ∈ {1, 2, 3, …}` all work with the same code. It also halves skip memory.

---

## Index of Public Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `SimpleTokenizer` | `AI/tokenizer.py` | Text ↔ ids |
| `build_default_tokenizer` | `AI/tokenizer.py` | Pre-built vocab |
| `DDPMScheduler` | `AI/scheduler.py` | Forward + reverse DDPM |
| `DDIMScheduler` | `AI/scheduler.py` | Deterministic reverse |
| `TextTransformerEncoder` | `AI/model.py` | Text → embeddings |
| `VideoUNet` | `AI/model.py` | Video noise predictor |
| `TinyVAEEncoder` / `TinyVAEDecoder` | `AI/model.py` | Latent compression |
| `TextToVideoModel` | `AI/model.py` | Full pipeline |
| `validate_dataset` / `repair_dataset` | `AI/dataset.py` | Dataset QA |
| `VideoTextDataset` | `AI/dataset.py` | PyTorch dataset |
| `collate_fn` / `random_prompt_dropout` | `AI/collate.py` | Batching + CFG prep |
| `train` / `_main` | `AI/train.py` | Training |
| `generate_video` / `_main` | `AI/inference.py` | Inference |
| `export_video` / `frames_from_video` | `AI/export.py` | FFmpeg I/O |
| `detect_profile` / `report` | `AI/resource_detector.py` | Hardware probe |
| `train_mvp` / `generate_mvp` | `AI/lightweight/*` | Fast MVP path |
| `app` / `main` | `app.py` | Flask service |
| `main` | `health_check.py` | Diagnostics |
