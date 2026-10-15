# Implementation Plan: Mechanistic Interpretability Audit for the ECG FILIP Model

**Audience:** Autonomous coding agent working in the existing ECG research repository  
**Status:** Implementation specification; not a claim that these experiments have been run  
**Primary objective:** Determine whether the ECG vision encoder learns clinically meaningful internal representations and whether the regions highlighted by FILIP actually influence downstream ECG predictions.  
**Guiding reference:** Goodfire, *Neural Networks Are Already Clean*, https://already-clean.goodfire.com/

---

## 0. Agent operating instructions

1. **Inspect the existing repository before changing anything.** Do not assume specific module names, checkpoint paths, image dimensions, patch sizes, or model output conventions. Search for the existing FILIP training/inference code, CLIP/ViT wrappers, evaluation pipeline, renderer, dataset loaders, and visualization scripts. Earlier work was organized under `/filip/`; use that location if it exists, but respect the repository's actual structure.
2. **Reuse the project's established model-loading code, splits, preprocessing, and evaluation metrics.** Do not construct an independent inference pipeline that silently changes image normalization, lead layout, image resolution, interpolation, patch ordering, or label definitions.
3. **Do not retrain the large encoders.** Analyses should use frozen checkpoints. A lightweight, controlled linear probe may be trained only if necessary to make comparisons valid and affordable.
4. **Keep the existing training pipeline unchanged.** Implement new analysis utilities in a separate package or directory; make no destructive checkpoint or dataset modifications.
5. **Work incrementally:** build a smoke test, validate the analysis against unchanged model outputs, then run small subsets before scaling. Log missing resources or unsupported architectures explicitly rather than inventing substitutes.
6. **Avoid unverified clinical claims.** Diagnostic labels and FILIP text similarity are not ground-truth labels for individual waveform features. Differentiate automated morphology proxies from cardiologist-validated annotations.
7. **Record reproducibility metadata:** Git commit, checkpoint hash, model/config identity, dataset version, sample IDs, split, RNG seed, transform settings, hook names, hardware, and output timestamps.
8. **Treat this as an interpretability audit, not a new state-of-the-art classifier experiment.** Preserve blinded test evaluations where possible; use validation data for exploration and freeze decisions before test evaluation.

---

## 1. Research questions and falsifiable hypotheses

### RQ1 — What do the vision encoder's internal units represent?

Explore ViT MLP units at the *patch-token level*. Determine whether highly activated patches exhibit repeatable ECG morphology (for example widened QRS, ST displacement, T-wave inversion) rather than simply gridlines, printed labels, borders, baseline position, or generic sharp lines.

- **H1:** A nontrivial subset of units have reproducible, reasonably specific morphology-associated response patterns on held-out ECGs.
- **Null/alternative:** Unit responses are not stable, are explained by rendering/position artifacts, or do not correspond to a defensible clinical concept.

### RQ2 — Are FILIP-highlighted patches causally important to predictions?

Compare the existing patch–report similarity map with controlled activation interventions at corresponding positions and with downstream diagnostic logit changes.

- **H2:** High-FILIP-similarity regions cause a larger class-specific logit decrease when ablated than matched random and low-similarity regions.
- **Null/alternative:** Similarity maps are not more causally predictive than controls, or effects can be explained by patch position, intervention size, or general model sensitivity.

### RQ3 — What changes after ECG–report pretraining?

Compare initial CLIP weights against the ECG–report pretrained vision encoder, and downstream-adapted checkpoints where comparable.

- **H3:** ECG–report pretraining yields more stable or selective morphology-associated responses and/or improved agreement between localization and intervention effects.
- **Null/alternative:** Raw CLIP is already similarly interpretable; gains are minimal or are attributable to different classifiers, preprocessing, architectures, or checkpoint selection.

**Important:** These hypotheses do **not** require accepting or rejecting superposition globally. The deliverable is evidence about this particular ECG model.

---

## 2. Existing project context — verify, do not hard-code

Known research setup from prior work:

- ECG images are deterministically rendered from 12-lead waveform recordings.
- Phase I: CLIP-initialized vision and text encoders, aligned using a FILIP-style patch–token late-interaction objective on ECG image–report pairs (MIMIC-IV-ECG).
- Phase II: downstream diagnosis prediction with the transferred vision encoder and a classification head; text branch removed for classification.
- Primary classification evaluation includes PTB-XL, with superclasses **NORM, MI, STTC, CD, HYP**. Other evaluated tasks/datasets may include PTB-XL subclasses/form/rhythm, CPSC, and CSN.
- There are potentially useful checkpoints for (a) raw CLIP initialization, (b) ECG–report pretrained model, and (c) downstream-finetuned model.
- Prior experiments used different image/model configurations; **never assume all saved checkpoints are directly comparable**.

Verify from code and metadata:

- Actual vision backbone variant, number of blocks, MLP activation (e.g. GELU/QuickGELU), layer normalization, token layout, and whether CLS exists.
- Image size, patch stride/size, actual patch grid, positional-embedding interpolation, patch order, treatment of CLS, and image transform details. Descriptions of "16×16" patches in old materials may refer to a display or configuration rather than every checkpoint's actual tokenizer.
- How Phase II produces logits (CLS, mean pooled spatial tokens, weighted pooling, etc.).
- Whether the model has a working image–report similarity-map API; whether the map is token-wise, report-wise, class-text-wise, normalized, etc.
- Which checkpoint variants have an attached head, and how existing evaluation is performed.
- Actual split files; for PTB-XL use the repo's canonical train/validation/test fold assignment (often folds 1–8 / 9 / 10, but verify).
- Ground-truth morphology annotations or waveform feature measurements available in the project, if any.

### Completion gate 0

Produce `outputs/interpretability/repo_audit.md` documenting:

1. Paths to loaders, checkpoints, renderer, split metadata, logits, FILIP maps.
2. The chosen **primary checkpoint** and **primary dataset/split**.
3. Hookable ViT modules and expected tensor shapes.
4. Which comparison checkpoints are truly architecture- and input-compatible.
5. Missing resources or constraints, and exact fallback decisions.

If no functional checkpoint and inference pipeline are available, stop analysis work and report the concrete missing requirements. Do not create fake experiments.

---

## 3. Proposed software structure

Follow existing repository conventions. If nothing suitable exists, use this logical structure under the existing FILIP project:

```text
filip/
  interpretability/
    __init__.py
    config.py
    model_adapter.py       # existing checkpoint/loading bridge, forward + logits
    data_adapter.py        # existing preprocessing/splits + ECG metadata
    token_geometry.py      # patch-token -> rendered image coordinates
    hooks.py               # capture/intervene on intermediate units
    extract_activations.py # streaming top/bottom unit activation mining
    characterize_units.py  # morphology analysis, confound checks
    filip_maps.py          # bridge to existing late-interaction similarity
    interventions.py       # activation patching/ablation protocols
    compare_checkpoints.py # controlled model comparison
    metrics.py             # paired statistics + patient-level bootstrap
    visualizations.py     # ECG overlays, unit galleries, curves
    report.py              # reproducible Markdown/CSV/JSON summaries
    cli.py                 # or thin CLI scripts if repository uses scripts/
  configs/
    interpretability.yaml
  tests/
    test_interpretability_*.py
outputs/
  interpretability/
    <run_id>/
      config.yaml
      environment.json
      repo_audit.md
      unit_topk.parquet
      neuron_characterization.csv
      intervention_per_case.parquet
      intervention_summary.csv
      checkpoint_comparison.csv
      figures/
      report.md
```

File/module names are suggestions, not a mandate. Prefer small, testable Python modules to a monolithic notebook. Do not introduce heavy libraries unless necessary; `torch`, `numpy`, `pandas`, `matplotlib`, and the project's existing stack should suffice for MVP. `captum` or `nnsight` are **optional** later.

### Core interfaces (adapt to actual code)

```python
class ECGModelAdapter:
    def load(self, checkpoint_path: str, config_path: str | None = None): ...
    def preprocess(self, batch): ...
    def predict_logits(self, batch) -> torch.Tensor: ...  # [B, num_classes]
    def vision_tokens(self, batch) -> torch.Tensor: ...    # [B, num_tokens, dim]
    def available_hook_sites(self) -> dict: ...
    def get_filip_patch_scores(self, batch, text_inputs=None): ...  # if supported

class PatchGeometry:
    def token_to_image_box(self, token_index: int) -> tuple[int, int, int, int]: ...
    def valid_patch_indices(self) -> list[int]: ...
```

Do not force an adapter method if the corresponding capability is unavailable. In particular, do not invent class-specific FILIP text prompts or implement a different FILIP scoring function and present it as the original method.

---

## 4. Work package A — Instrument and validate existing inference

**Goal:** Measure internal activations without changing model behavior.

### Tasks

1. Load one existing classifier checkpoint using the repo's established code in `eval()` mode.
2. Reproduce logits and predicted class labels for a fixed mini-batch using both existing inference and the new adapter. Assert numerical agreement within appropriate precision tolerance.
3. Enumerate ViT blocks and identify hook targets:
   - Prefer the **MLP expansion activation** per transformer block (pre-nonlinearity and post-nonlinearity when available).
   - Also expose relevant block outputs / patch-token embeddings for later region interventions.
   - **Do not automatically interpret an arbitrary residual-stream channel as a discrete neuron.**
4. Use forward hooks for recording; use safe forward hooks or explicit module wrappers for changing an activation during intervention. Do not mutate persistent weights or register duplicate hooks.
5. Confirm tensor conventions: `[batch, tokens, channels]` vs alternatives, special tokens, masked tokens, and layer IDs.
6. Implement patch/token-to-image geometry using actual model config and transform metadata. Handle resize, interpolation and crops explicitly.
7. Render a sanity-check ECG with an overlay for *every corner token* and one center token. Manually/automatically verify indexing, shape, and alignment.
8. Include no-hook and passive-hook comparison: outputs must match baseline; repeated runs under fixed seed must be stable.

### Tests

- Same image -> same tensor and logits with the existing and new pipeline.
- Hook activation shape agrees with inspected architecture.
- Spatial token count matches computed patch grid after subtracting special tokens.
- Token `i` maps to the correct actual image region; CLS never receives a fake image box.
- Hooks are removed after use and do not leak state into later runs.

### Gate A

A smoke test on **8–16 ECGs** runs end-to-end, creates a correct overlay, captures intermediate activations, and reproduces baseline logits exactly to numerical tolerance.

---

## 5. Work package B — Discover high-activating ECG patches

**Goal:** Build an interpretable gallery of original-model MLP units, without fitting an SAE.

### Sampling and storage

- Start with **one architecture/checkpoint**, preferably the finished ECG–report pretrained or downstream model with stable inference.
- Select **2–4 separated transformer blocks** (early, middle, late) and **32–128 sampled/targeted MLP units per block** for the pilot. Units and layers must be configurable.
- Pilot on **~200–500 unique ECG recordings**, stratified by available diagnosis labels and patient ID as much as feasible. Scale after confirming compute and useful signal.
- Always use a stable `study_id`/sample ID and `subject_id` or patient grouping when available.
- For each sample/unit, calculate meaningful patch-level scores (e.g. maximum **post-activation** over non-CLS tokens) and preserve token coordinates. Keep preactivation scores separately to study negative responses.
- Maintain a **streaming top-K and bottom-K** set per unit (default `K=20`) rather than persisting every activation tensor for every sample.
- Optionally store a small, reproducible reservoir of non-extreme activations to check specificity.

### Required outputs

- `unit_topk.parquet`: checkpoint, layer, unit, sample ID, token index, pre/post scores, rank, image box, label metadata, preprocessing version.
- A gallery for selected units: top 20 patches **with surrounding ECG context**, full ECG with highlighted boxes, and low/negative preactivation cases where applicable.
- Aggregate statistics: activation quantiles, concentration by lead/row/column/patch location, between-subject repeatability, diagnosis distribution, and activation sparsity.

### Confound checks

- Compare high-activation patches with patches at matched grid locations from other ECGs.
- Look for gridline density, printed lead labels, rendering boundaries, rhythm-strip position, black trace thickness, padding, and image brightness.
- Avoid declaring a unit "MI" or "ST elevation" merely because it activates in positive-diagnosis examples.
- Distinguish **morphology association** from classification correlation and causal contribution.

### Gate B

The agent can output a gallery for **at least 20 selected units** spanning more than one layer, trace each thumbnail to an exact sample/patch coordinate, and report artifacts/unknowns without inventing medical explanations.

---

## 6. Work package C — Characterize candidate morphology-associated units

**Goal:** Test possible interpretations beyond visual pattern matching.

### Candidate concepts (examples, not automatic ground truth)

- QRS width / widening
- ST-segment displacement
- T-wave inversion or polarity
- Q-wave morphology
- R-peak / QRS complex morphology
- Heart rhythm regularity **only if the context/window suffices**
- Lead-specific patterns, including bundle branch block morphology where labels and context support it

### Annotation hierarchy

1. Prefer **existing waveform-derived measurements or clinically reviewed feature annotations**, with unit/lead definitions documented.
2. If those are unavailable, implement a separately flagged *proxy* based on robust signal processing and report limitations; use original digital waveform and sample rate, not just pixel heuristics when available.
3. Diagnose-level PTB-XL superclass labels are only **weak labels** for morphology, not proof that a specific patch contains that finding.
4. Free-form MIMIC reports may guide qualitative examples, but report phrases are not automatically spatially grounded morphology labels.

### Validation protocol

For a preselected candidate unit/concept:

1. Form an explicit, falsifiable unit-description hypothesis (e.g. "responds to wide QRS complexes, independent of printed lead name").
2. Gather positive, negative, and hard-negative ECG examples from **different patients**. Include morphology-matched or position-matched controls where feasible.
3. Measure concept discrimination using held-out samples (AUROC/AUPRC for supported binary features; correlation for continuous measurements such as QRS duration).
4. Evaluate whether responses generalize across leads/positions or are lead-specific; neither pattern is intrinsically better, but report it.
5. Test preactivation-negative examples separately; negative preactivation does **not** imply negative causal influence after GELU/QuickGELU.
6. Human-review a small blinded, representative gallery when possible. Do not score interpretability solely by how plausible an LLM-generated caption sounds.

### Outputs

A table with `checkpoint`, `layer`, `unit`, `hypothesis`, `annotation_source`, `n_subjects`, `validation_metric`, `CI`, `confound_result`, and one of `supported`, `not_supported`, `inconclusive`.

### Gate C

At least **5–10 candidate units** have explicit held-out tests, and the report clearly distinguishes visually appealing observations from validated results. If reliable morphology labels are unavailable, ship a *qualitative gallery plus weak-label analysis*, explicitly without claiming clinical unit specificity.

---

## 7. Work package D — Causal test of FILIP heatmaps (highest priority)

**Goal:** Evaluate whether FILIP-highlighting corresponds to causal importance for downstream diagnosis predictions.

### Required model-specific clarification

The Phase I FILIP map uses text-to-patch or patch-to-text alignment; the Phase II classifier may use only image features. A similarity score is **not** itself a class-specific causal attribution. Precisely document which text was used, which image features, how token similarities are pooled, and which final diagnosis logit is tested.

### Protocol

1. For each held-out ECG sample, compute diagnosis logits from the original downstream head.
2. Retrieve existing FILIP patch similarity scores, using the **actual pretrained image/text encoders and original FILIP logic**. If no such function exists, implement a faithful reproduction and verify it against saved prior visualizations on several known samples.
3. Convert similarity scores to spatial patch indices in the same rendered image reference frame as the downstream model. If models use different grids/image sizes, map to common pixel regions rather than equating token indices.
4. Define high-similarity regions (e.g. top 5%, 10%, 20% of valid patches), corresponding bottom/low regions, and matched random controls. For fair comparisons, keep **the same number/area of patches** across conditions.
5. Prefer **activation-level interventions** at a hook site with a valid causal path to the downstream head:
   - `zero`: replace targeted activations with zero.
   - `mean`: replace with per-layer mean activation estimated only on a separate training/calibration subset.
   - `matched_swap` (optional): replace with activation from another ECG matched by position and broad rendering characteristics.
6. Propagate through the **remaining** model and capture changed logits. Do not simply modify an activation tensor after the last point through which the classification head could read it.
7. Preserve unchanged outputs for non-intervention baselines. Do not physically erase pixels or crop the ECG as the primary ablation; such edits can generate out-of-distribution artifacts. Pixel masks may be exploratory controls only.
8. For each target class `c`, compute:

   `delta_logit_c = logit_c(original) - logit_c(intervened)`

   Large positive values indicate that the removed/replaced activations supported the class score **under the chosen intervention**, not necessarily that the corresponding waveform is clinically diagnostic.
9. Include both **predicted-label and ground-truth-label analyses** where appropriate, since they answer different questions. For multilabel ECG datasets, do not silently reduce to single-label accuracy.
10. Sweep fraction of ablated regions and plot logit change vs ablated fraction. Repeat with a few random seeds for random controls.
11. Optionally measure the change in macro-AUROC/micro-AUROC across the held-out cohort, using the repository's exact multilabel evaluation convention.

### Controls and validity checks

- Same-sized random patch sets, repeated with multiple seeds.
- Low-FILIP-similarity patch sets.
- Position-matched random patches (important for fixed 12-lead grid layout).
- Equal intervention type/strength for all comparisons.
- Non-target class logits to distinguish class-specific effects from indiscriminate prediction collapse.
- A no-op intervention must reproduce the original logits.
- A positive test intervention on an intentionally influential feature must measurably affect downstream activations/logits; otherwise the selected hook site is invalid.
- If FILIP text is an actual patient report, make clear that the map is potentially conditioned on information **not available to a prospective classifier**. Prefer report-independent class descriptions as an additional separately named analysis; do not conflate these with the original FILIP objective.

### Statistical analysis

- **Unit of resampling:** patient/subject (not patches, and not correlated ECGs treated as independent).
- Report paired mean/median `delta_logit`, high-minus-random and high-minus-low differences, subject-bootstrap 95% CIs, and an effect-size summary.
- Preserve per-class results (NORM, MI, STTC, CD, HYP where supported) as well as macro summaries.
- If testing many classes/units/comparisons, report exploratory status and use an appropriate multiplicity correction or predefined primary comparison.

### Required outputs

- `intervention_per_case.parquet`: patient/sample IDs, checkpoint IDs, class, source-map type, patch set, region size, layer/hook, intervention type, original and intervened logits, seed.
- `intervention_summary.csv`: paired estimates and uncertainty.
- Figures: average effect by patch selection strategy; dose–response curves; at least 10 interpretable image overlays contrasting high, low, and random regions.
- An explicit conclusion: **supports**, **does not support**, or **inconclusive** for H2, with reasons.

### Gate D

The pipeline produces reproducible paired intervention results on an initial **~100 held-out ECGs**, passes no-op and causal-path tests, and only then expands to a pre-registered or frozen evaluation subset large enough for robust CIs.

---

## 8. Work package E — Controlled checkpoint comparison

**Goal:** Decide whether ECG image–report pretraining changes interpretability, avoiding misleading raw-CLIP vs adapted comparisons.

### Checkpoints

- `A`: original CLIP vision encoder.
- `B`: vision encoder after ECG–report alignment pretraining.
- `C`: downstream-adapted encoder (if available and appropriate).

### Rules for fair comparisons

1. Use **the same ECG sample IDs and canonical rendered images** wherever architectures permit.
2. Never compare individual unit indices across models as if their semantic identity were preserved after training. Neurons can rotate, split, merge, or change functions. Compare population statistics and, optionally, activation-matched units on an independent calibration split.
3. Compare hidden representations at **matched depths and compatible architecture settings**; if backbones differ, report that these are not matched neuron comparisons.
4. For **prediction-logit-based** causal comparisons, ensure each encoder has a valid classifier head trained/evaluated under a controlled protocol. Preferred: frozen encoders with identically configured and identically trained lightweight linear probes, same train/validation/test patients, optimizer/epochs/seeds/model-selection rules.
5. The final downstream-finetuned head can be analyzed **within its own model** even if a raw CLIP baseline lacks a comparable head. Do not compare its absolute logit shifts against an incompatible head and attribute differences to pretraining.
6. Separate effects of multimodal pretraining from effects of downstream finetuning.
7. Restrict probe training to a modest run; do not retrain the foundation encoder. If controlled probe training cannot fit available resources, report a representation-only comparison instead.

### Comparable metrics

- Fraction of candidate units with confirmed morphology association under the same labeling protocol.
- Median held-out concept AUROC / correlation, with subject-bootstrap CIs.
- Activation stability across patients and rendering perturbations (if appropriately defined).
- Degree of FILIP localization vs intervention agreement, where map computation is valid.
- Downstream AUROC for controlled linear probes as context, **not** as an interpretability metric.

### Gate E

A comparison table explicitly indicates which pairs are *controlled*, *descriptive only*, or *not comparable*. No causal claim that FILIP created a clinical feature is made without appropriate evidence.

---

## 9. Optional work package F — Deeper circuit analysis (defer until MVP works)

Only attempt after packages A–E are usable:

- **Unit-level causal interventions:** zero/mean replacement of selected post-activation MLP units across selected patch positions, tracking class-logit changes.
- **Activation patching:** transplant activations from a matched normal ECG into an abnormal ECG (or vice versa) at a chosen internal node. Carefully separate changes in morphology from other differences.
- **Attribution tracing:** use gradients, integrated gradients, or activation tracing to identify earlier contributors to validated late-layer units. `captum` can help; `nnsight` may help with nested interventions if compatible with the installed model.
- **Waveform-valid counterfactuals:** modify the *digital ECG recording* under medically defensible constraints, rerender deterministically, and test whether expected units change. Never interpret unrealistic synthetic waveforms as clinical evidence.
- **Activation maximization / Lucent:** exploratory only; image-space optimization on ECGs is especially prone to generating meaningless grid/line artifacts. Prefer constrained waveform-space tests.
- **Cross-dataset generalization:** evaluate validated hypotheses on CSN/CPSC or other held-out sources only after verifying acquisition, label mapping, rendering, and distribution differences.

**Do not** make SAE training, activation maximization, automated agent-written neuron labels, or end-to-end circuit maps prerequisites for the main study.

---

## 10. Suggested configuration

Provide an editable YAML with actual defaults established in the repository audit. A *schema example*, not a guaranteed usable path:

```yaml
analysis:
  seed: 42
  device: cuda
  output_root: outputs/interpretability
  precision: fp32                 # favor numerical clarity for initial validation
  pilot: true

model:
  primary_checkpoint: null        # discover and fill from repository
  model_config: null
  filip_checkpoint: null          # required only for original FILIP maps
  comparison_checkpoints: []

# Paths and split names must match the repository.
data:
  dataset: ptbxl
  split: validation
  patient_group_key: subject_id
  max_unique_records: 300
  class_names: [NORM, MI, STTC, CD, HYP]

units:
  layers: []                      # populate actual valid block IDs
  units_per_layer: 64
  top_k: 20
  include_preactivation: true

intervention:
  target_layers: []
  fraction_grid: [0.05, 0.10, 0.20]
  strategies: [high_filip, low_filip, random, position_matched_random]
  replacement: [zero, mean]
  random_seeds: [42, 43, 44, 45, 46]
  target_classes: [NORM, MI, STTC, CD, HYP]

statistics:
  bootstrap_samples: 1000
  bootstrap_unit: patient
  ci_level: 0.95
```

The agent should fill real paths and layer indices after inspecting the repo. If a field is unsupported, reject the configuration with an informative error.

### CLI targets

These are conceptual contracts; implement equivalent commands consistent with the repository:

```bash
python -m filip.interpretability.cli audit --config configs/interpretability.yaml
python -m filip.interpretability.cli smoke-test --config configs/interpretability.yaml
python -m filip.interpretability.cli extract-units --config configs/interpretability.yaml
python -m filip.interpretability.cli characterize --config configs/interpretability.yaml
python -m filip.interpretability.cli intervene --config configs/interpretability.yaml
python -m filip.interpretability.cli compare --config configs/interpretability.yaml
python -m filip.interpretability.cli report --config configs/interpretability.yaml
```

If `filip` is not an importable top-level package, choose a valid alternative instead of introducing brittle Python path hacks.

---

## 11. Testing and engineering quality

### Unit tests

- Token-to-image-box transforms for known grid shapes, including CLS offsets and non-square grids.
- Streaming top-K/bottom-K selection matches a small brute-force reference.
- Unit extraction excludes special tokens and records correct patient IDs.
- Bootstrap sampling is at the patient level and reproducible under a fixed seed.
- Paired intervention metrics and class indexing have known expected results on synthetic tensors.
- Hook cleanup on exceptions and repeated model calls.
- All CLI configurations validate before expensive computation starts.

### Integration tests

- New baseline inference equals existing evaluation for a small fixed batch.
- Passive hooks do not change logits.
- No-op intervention does not change logits.
- Perturbations at chosen hook site propagate to later layers and can affect logits.
- Saved activation galleries correctly highlight the original image pixel area.
- FILIP score reproduction, if implemented, agrees with original pipeline outputs on fixed samples.
- Exported tables have no unexpected duplicate IDs, missing class columns, or incorrect split mixing.

### Resource safeguards

- No full-dataset activation dumps by default; use streaming extraction and narrow selected layers/units.
- Batch sizes auto-configurable; support CPU smoke tests and single-GPU pilot.
- Catch CUDA out-of-memory with useful guidance; do not silently downgrade precision in experiments that are being compared.
- Resume lengthy extraction/intervention runs without duplicating samples.
- Save progress periodically and retain run manifest/hash checks.

### Privacy and data handling

- Keep ECG images, MIMIC reports, sample IDs, and derived patient data in approved local research storage. Avoid embedding free-text patient information or identifiable metadata into public reports.
- Generate publication-safe aggregate tables/figures separately from internal galleries.

---

## 12. Ordered implementation milestones and deliverables

| Milestone | Implementation | Deliverable | Move on when… |
|---|---|---|---|
| **M0: Repository audit** | Discover the actual architecture, checkpoints, splits, renderer and FILIP scoring | `repo_audit.md` | A valid, reproducible analysis target is established |
| **M1: Instrumentation** | Model/data adapters, MLP hooks, token geometry, baseline verification | Smoke-test report + overlay | Original inference agrees numerically; mappings are correct |
| **M2: Neuron gallery** | Streaming activation mining and galleries | Top-K data + 20+ unit galleries | Candidate units can be examined without obvious indexing errors |
| **M3: Causal heatmap test** | Controlled region intervention and paired statistics | Per-case table, effect plots, bootstrap CIs | No-op/causal-path controls pass and H2 can be evaluated |
| **M4: Unit hypothesis tests** | Morphology-based candidates, hard negatives, confound tests | Unit characterization table | Evidence quality and uncertainty are explicit |
| **M5: Checkpoint comparisons** | Controlled raw-CLIP vs aligned vs adapted analysis | Fair-comparison table | Head/preprocessing confounds are addressed |
| **M6: Final report** | Synthesize plots, tests, interpretations, failure cases | `report.md` + publication-quality plots | Results are reproducible and limitations explicit |

**Priority order if time or compute is limited:** M0 → M1 → M3 → M2 → M4 → M5 → M6.  
This keeps the clinically and scientifically most important causal question ahead of extensive neuron cataloging. M2's pilot extraction can run before or alongside M3, but it is not required for region-level intervention tests.

---

## 13. Success, failure, and reporting standards

### Minimum viable outcome

An independently rerunnable experiment on the existing ECG classifier that shows:

1. The original FILIP map for a held-out ECG.
2. The same ECG under high-similarity, low-similarity, and matched-random patch activation interventions.
3. Changed class logits under each intervention.
4. Patient-bootstrap uncertainty over a held-out sample set.
5. Validated image/patched-token alignment and causal-path checks.

**This is worthwhile even if the result is negative.** A failure of FILIP maps to outperform controls would be an important finding about explanation faithfulness, not an implementation failure.

### Stronger scientific outcome

- Several reproducible, morphology-associated native ViT units with valid feature-specific evidence.
- A controlled demonstration that ECG–report pretraining changes their specificity, stability, or causal relevance.
- Evidence that the patch–text maps track model reliance, with appropriate baselines and sensitivity analysis.

### Final report sections

1. Exact research questions and hypotheses.
2. Checkpoints/data/splits/renderer and model-hook specifications.
3. Validation of inference equivalence and coordinate mapping.
4. Representative activation galleries (with confound controls).
5. FILIP-high vs low/random intervention results, per class and aggregate.
6. Patient-level confidence intervals and sensitivity to intervention type/size/layer.
7. Raw CLIP vs ECG-pretrained vs adapted comparisons, clearly separated by valid comparison status.
8. Negative cases and failure analysis.
9. What is causal, what is merely correlational, and what is clinically validated.
10. Reproduction commands, configuration, compute notes, and provenance.

### Claims not permitted without further evidence

- "The model learned an ST-elevation neuron" based only on representative patches.
- "FILIP attention proves the diagnosis location" based only on similarity maps.
- "Superposition does not exist" based on a few interpretable units.
- "The ECG–report pretraining caused better interpretability" from comparisons that change classifier heads, backbones, splits, or preprocessing without controls.
- "The explanation is clinically correct" without suitable morphology ground truth or expert evaluation.

---

## 14. Instructions for the coding agent's final handoff

At the end of implementation, provide:

- Brief inventory of created/modified files.
- Exact example CLI commands, using verified paths/configs.
- Results of smoke tests, unit tests, and intervention sanity checks.
- The actual number of ECGs/patients, model checkpoints, and runs analyzed.
- Links/paths to produced HTML/PNG/CSV/Parquet/Markdown artifacts.
- A clear list of completed milestones vs deferred features.
- The main result in one paragraph, **without turning tentative interpretations into medical claims**.
- Any architecture or dataset blockers, described precisely.

**Agent instruction:** Prioritize a correct, reproducible, causal validity test over elaborate visualization or broad feature discovery. Do not substitute fabricated metrics, simulated model outputs, or unrelated public ECG images for outputs from the actual project checkpoint.
