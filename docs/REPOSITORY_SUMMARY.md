# GEM Repository Technical Summary

This document summarizes the repository structure, core components, data flow, and implementation details of this GEM fork. It is intended as a compact but detailed reference for comparing this repository with another codebase.

## 1. Project identity and purpose

GEM is an ECG-focused multimodal large language model (MLLM) repository built on a LLaVA-style codebase. The project goal is grounded ECG understanding: combine 12-lead ECG waveform time series, rendered ECG paper images, and text instructions/reports so a language model can produce clinically grounded interpretations that cite waveform evidence. The repository combines four major systems:

1. **GEM/LLaVA model stack** in `llava/`: language-model wrappers, multimodal input assembly, training, inference, serving, and generic LLaVA evaluation utilities.
2. **ECG-CoCa encoder stack** in `ecg_coca/`: an OpenCLIP/CoCa-derived ECG-text contrastive/captioning model used as the ECG time-series tower.
3. **ECG image/data generation and prompt tooling** in `gem_generation/`: ECG paper image synthesis and instruction-generation prompts.
4. **Benchmark and GPT-based evaluation tooling** in `evaluation/` and `gem_evaluation/`: ECG-Bench, ECG-Grounding, report scoring, baseline inference, and GPT judge prompts.

The package metadata still identifies the Python project as `llava`, version `1.2.2.post1`, with PyTorch, Transformers, PEFT, bitsandbytes, Gradio/FastAPI, timm, and related LLaVA dependencies declared in `pyproject.toml`.

## 2. Top-level repository layout

| Path | Role |
| --- | --- |
| `README.md` | User-facing GEM overview, setup, data layout, training/evaluation commands, external model/data links, and citation. |
| `pyproject.toml` | Python packaging metadata and core runtime/training dependency pins. |
| `setup.sh` | Environment setup entry point referenced by the README. |
| `predict.py`, `cog.yaml` | Cog/Replicate-style predictor wrapper and deployment metadata. |
| `llava/` | Main model/training/evaluation/serving code inherited from and modified from LLaVA. |
| `ecg_coca/` | ECG-CoCa/open_clip implementation and training utilities for the ECG time-series encoder. |
| `gem_generation/` | ECG image generator and generation prompt templates for grounded instruction data. |
| `evaluation/` | GEM/ECG benchmark scripts, ECG-Bench metrics, report evaluation, and baseline inference wrappers. |
| `gem_evaluation/` | GPT-based grounding evaluation script, prompts, and notebooks for result post-processing. |
| `scripts/` | GEM training shell script, DeepSpeed config, and many inherited LLaVA finetuning/evaluation helper scripts. |
| `pics/` | README figures and logo assets. |

## 3. Runtime/data expectations

The README expects external data under `./data` split into:

- `ecg_timeseries/`: WFDB or dataset-specific source ECGs from MIMIC-IV-ECG, PTB-XL, CODE-15, CPSC 2018, CSN, G12E, etc.
- `ecg_images/`: rendered ECG image folders, including generated MIMIC/PTB-XL images and ECG-Instruct/ECG-Bench assets.
- `ecg_bench/`: ECG-Bench image assets and ECG-Grounding test JSONs.
- `ecg_jsons/`: instruction data such as `ECG_Grounding_30k.json`.

GEM also requires a pretrained ECG-CoCa checkpoint (`cpt_wfep_epoch_20.pt`) and a base MLLM such as PULSE or LLaVA. A key operational note in the README is that Hugging Face GEM-7B configs must point `mm_ecg_tower` at the local ECG-CoCa checkpoint before inference.

## 4. Core GEM/LLaVA model stack (`llava/`)

### 4.1 Constants and prompt/conversation primitives

- `llava/constants.py` defines the special token protocol used by multimodal examples:
  - `IGNORE_INDEX = -100` for masked labels.
  - `IMAGE_TOKEN_INDEX = -200` as the sentinel ID replaced by multimodal embeddings.
  - `<image>`, `<im_patch>`, `<im_start>`, `<im_end>`, and `<image-placeholder>` strings.
- `llava/conversation.py` contains LLaVA/FastChat conversation templates and separator styles used by training and evaluation scripts.
- `llava/mm_utils.py` provides image-token tokenization and image preprocessing helpers, including any-resolution image support used by the LLaVA multimodal path.

### 4.2 Multimodal architecture integration (`llava/model/llava_arch.py`)

The central GEM-specific architecture logic is in `llava/model/llava_arch.py`.

#### `LlavaMetaModel`

`LlavaMetaModel` conditionally attaches two modality towers when corresponding config fields exist:

- `mm_ecg_tower` triggers construction of `self.ecg_tower` plus `self.ecg_projector`.
- `mm_vision_tower` triggers construction of `self.vision_tower` plus `self.mm_projector`.

Initialization methods:

- `initialize_ecg_modules(model_args, fsdp=None)`:
  - stores `mm_ecg_tower` and `open_clip_config` in model config;
  - builds or loads the ECG tower;
  - sets `use_mm_proj = True`;
  - builds an ECG projector with `ecg_projector_type` (default `linear`);
  - optionally loads `pretrain_ecg_projector` weights by extracting keys containing `ecg_projector`.
- `initialize_vision_modules(model_args, fsdp=None)`:
  - builds or loads the CLIP vision tower;
  - records vision projector type, hidden size, selected layer/feature, and patch merge mode;
  - supports LLaVA's `unpad` image-newline parameter for spatial any-resolution patches;
  - optionally loads pretrained `mm_projector` weights.

#### `LlavaMetaForCausalLM`

The causal-LM mixin performs feature extraction and input embedding replacement.

Key details:

- `encode_ecgs(ecgs)` passes waveforms through the ECG tower, then through `ecg_projector`, and then through the shared image/LLM projector `mm_projector`. This means ECG features are first normalized to the vision/projector hidden dimension expected by GEM, then mapped into the LLM embedding dimension.
- `encode_images(images)` passes image tensors through the vision tower and `mm_projector`.
- `prepare_inputs_labels_for_multimodal(...)` is the core fusion method. If no image tower/images exist or a one-token generation step is being processed, it returns normal text input. Otherwise it:
  1. ensures a single ECG shaped `12 x 5000` is batched;
  2. encodes ECG features;
  3. encodes image features, including support for list/5-D batched any-resolution crops and spatial patch merging;
  4. concatenates ECG and image token features, either per sample for split images or along token dimension for standard batches;
  5. removes text padding using the attention mask;
  6. locates `IMAGE_TOKEN_INDEX` in the tokenized prompt;
  7. replaces each `<image>` placeholder with the concatenated multimodal ECG+image feature sequence;
  8. masks all inserted multimodal feature positions with `IGNORE_INDEX` labels;
  9. truncates to the tokenizer max length and pads embeddings/labels/attention masks back to a batch.

A subtle repository-specific implementation detail is that the ECG features are inserted at the same placeholder position as image features: there is no separate ECG sentinel token. The data and evaluation prompts use the existing `<image>` token to indicate insertion of the concatenated ECG/image embedding block.

### 4.3 Modality towers (`llava/model/multimodal_encoder/`)

`builder.py` exposes:

- `build_ecg_tower(cfg, **kwargs)`: resolves the ECG tower path from `mm_ecg_tower` or `ecg_tower`. It only accepts a path that exists on disk and constructs a `CLIPECGTower`.
- `build_vision_tower(cfg, **kwargs)`: accepts local paths or known CLIP/OpenAI/LAION/ShareGPT4V identifiers and returns either `CLIPVisionTower` or `CLIPVisionTowerS2`.

`clip_encoder.py` contains the tower wrappers:

- `CLIPECGTower`:
  - imports `get_ecg_encoder` from `ecg_coca.training`;
  - requires an `open_clip_config` such as `coca_ViT-B-32`;
  - loads the ECG-CoCa ECG submodule from the checkpoint path;
  - freezes the ECG tower with `requires_grad_(False)`;
  - exposes hidden size, sequence length, patch size, device, dtype, and number of patches from the ECG encoder config;
  - `forward()` accepts a batch or list of ECG tensors and calls the underlying ECG encoder with `output_last_transformer_layer=True`, returning token-level features.
- `CLIPVisionTower`:
  - wraps Hugging Face `CLIPVisionModel` and `CLIPImageProcessor`;
  - freezes the vision encoder;
  - selects either patch tokens (`[:, 1:]`) or class+patch tokens from a configured hidden layer.
- `CLIPVisionTowerS2` extends CLIP vision encoding with multi-scale/S2 support.

### 4.4 Projectors (`llava/model/multimodal_projector/builder.py`)

The projector builder supports three families:

- `linear`: a single `nn.Linear`.
- `mlpNx_gelu`: an N-layer MLP with GELU activations.
- `identity`: pass-through `IdentityMap`.

Implementation details:

- `build_ecg_projector()` currently hardcodes the ECG input width as `768` and outputs `config.mm_hidden_size`.
- `build_vision_projector()` maps from `config.mm_hidden_size` to `config.hidden_size`.
- Because `encode_ecgs()` applies both `ecg_projector` and `mm_projector`, the ECG path is effectively `768 -> mm_hidden_size -> LLM hidden_size`.

### 4.5 Language-model wrappers (`llava/model/language_model/`)

The repository supports multiple LLaVA-style LM backbones:

- `llava_llama.py`: LLaMA/Vicuna-style causal LM wrapper.
- `llava_qwen.py`: Qwen2 causal LM wrapper.
- `llava_mistral.py`: Mistral wrapper.
- `llava_mpt.py`: MPT wrapper.

`llava_qwen.py` is especially relevant because it defines:

- `LlavaQwenConfig`, registered with model type `llava_qwen`.
- `LlavaQwenModel`, combining `LlavaMetaModel` with `Qwen2Model`.
- `LlavaQwenForCausalLM`, combining `Qwen2ForCausalLM` with `LlavaMetaForCausalLM`.

Its `forward()` method calls `prepare_inputs_labels_for_multimodal()` whenever `inputs_embeds` is not supplied, then delegates to the parent causal LM. Its `generate()` method similarly precomputes multimodal embeddings and passes `inputs_embeds` to Hugging Face generation; it disallows user-provided `inputs_embeds`.

### 4.6 Model loading (`llava/model/builder.py`)

`load_pretrained_model()` handles local/HF model loading for LLaVA/GEM or plain language models. Important behavior:

- Supports `load_8bit`, `load_4bit`, default FP16, and optional FlashAttention 2.
- Treats model names containing `llava` or `gem` as multimodal.
- Supports LoRA loading/merging when `model_base` is provided.
- Loads LLaMA, MPT, Mistral, Qwen, and Qwen-MoE wrappers depending on the model name.
- Adds multimodal special tokens to the tokenizer and resizes token embeddings.
- Loads and optionally moves the ECG tower and vision tower, returning `tokenizer, model, image_processor, context_len`.

One implementation quirk is that the Qwen branch references local variables such as `overwrite_config` and `attn_implementation` inside the branch; the non-overwrite branch sets them just before use for regular Qwen, but this area is brittle and should be compared carefully against any upstream/source repository.

## 5. Training stack (`llava/train/` and `scripts/`)

### 5.1 Main training script (`llava/train/train.py`)

`llava/train/train.py` is the main supervised fine-tuning entry point.

Argument dataclasses:

- `ModelArguments` adds GEM-specific fields:
  - `ecg_tower`, `open_clip_config`, `checkpoint_path`, `ecg_projector_type`, `pretrain_ecg_projector`;
  - vision tower/projector settings inherited from LLaVA;
  - image start/end and patch token options.
- `DataArguments` adds:
  - `image_folder`, `image_aspect_ratio`, `ecg_folder`, `ecg_seq_length` (default 5000).
- `TrainingArguments` adds:
  - quantization controls (`bits`, `double_quant`, `quant_type`), LoRA settings, multimodal projector LR, modality-length grouping, and attention implementation.

Preprocessing supports multiple conversation templates (`plain`, LLaMA-2, Vicuna v1, MPT, Qwen). Labels from human/instruction portions are masked with `IGNORE_INDEX` so loss is only computed on assistant responses.

`LazySupervisedDataset` expects JSON examples with at least:

- `conversations`: alternating human/GPT messages;
- `image`: relative ECG image path;
- `ecg`: relative WFDB ECG record path.

For each multimodal example it:

1. reads the waveform with `wfdb.rdsamp()` from `ecg_folder`;
2. replaces NaN/Inf with zero;
3. transposes to channel-first `12 x length` tensor;
4. pads or truncates to `ecg_seq_length` (default `5000`);
5. opens the image from `image_folder` as RGB;
6. applies one of `pad`, `anyres`, or `ori` image preprocessing modes;
7. preprocesses the conversation and returns `input_ids`, `labels`, `ecg`, `image`, and `image_size`.

`DataCollatorForSupervisedDataset` pads token IDs and labels, builds the attention mask, stacks ECG tensors when shapes match, stacks/list-preserves images, and carries `image_sizes` for any-resolution handling.

Training flow:

1. Parse HF arguments.
2. Load a multimodal wrapper if `vision_tower` is provided; otherwise load plain LLaMA.
3. Optionally quantize with bitsandbytes.
4. Optionally freeze the backbone.
5. Optionally enable gradient checkpointing.
6. Optionally add PEFT LoRA adapters, excluding multimodal modules from target linear layers.
7. Load tokenizer and conversation template.
8. Initialize ECG and vision modules and move towers to FP16/BF16.
9. Optionally tune only `mm_projector` and `ecg_projector` (`tune_mm_mlp_adapter`).
10. Optionally freeze both multimodal projectors.
11. Create `LLaVATrainer` and train from scratch or resume checkpoint.
12. Save LoRA state/non-LoRA trainables or full/adaptor weights.

### 5.2 Trainer and scripts

- `llava/train/llava_trainer.py` extends Hugging Face training logic with LLaVA-specific sampling and optimizer grouping, including projector learning-rate handling.
- `llava/train/train_mem.py` and `train_xformers.py` are memory/attention variants.
- `scripts/train_gem.sh` is the GEM training launcher the README tells users to edit with local paths.
- `scripts/zero2.json` is the DeepSpeed ZeRO-2 configuration used by training scripts.
- `scripts/llava_scripts/` contains inherited LLaVA pretraining, finetuning, LoRA, QLoRA, evaluation, conversion, and submission helpers.

## 6. ECG-CoCa stack (`ecg_coca/`)

`ecg_coca/` is a local OpenCLIP-derived implementation adapted to ECG time series and ECG-text alignment.

### 6.1 Model implementation (`ecg_coca/open_clip/`)

Important files:

- `coca_model.py`: defines `CoCa`, which includes text, ECG, and multimodal text-decoder towers. It can encode ECG and text independently and compute both contrastive logits and captioning logits when text is supplied.
- `model.py`: defines CLIP/ECG tower configuration dataclasses and builders such as `_build_ecg_tower()` and `_build_text_tower()`.
- `transformer.py`: transformer blocks, attention, residual blocks, and multimodal decoder implementation.
- `loss.py`: contrastive and captioning loss functions for CLIP/CoCa training.
- `factory.py`: model config discovery/creation functions such as `create_model_and_transforms()` and `get_model_config()`.
- `tokenizer.py` and `bpe_simple_vocab_16e6.txt.gz`: OpenCLIP-style tokenizer assets.
- `augmentations/`: ECG-specific augmentations such as baseline wander, cutmix, and random masking.
- `model_configs/coca_ViT-B-32.json`: the default config referenced by GEM training/evaluation scripts.

`CoCa.forward()` returns ECG features alone when text is absent; when text is present it encodes both modalities, computes contrastive logits, and runs a multimodal decoder for caption/generative loss.

### 6.2 ECG encoder loading (`ecg_coca/training/main.py`)

`get_ecg_encoder(model_name, checkpoint_path, device)` is the bridge used by GEM's `CLIPECGTower`. It:

1. obtains the OpenCLIP model config;
2. creates the full CoCa model and preprocessing transforms;
3. selects only `model.ecg`;
4. loads checkpoint keys beginning with `module.ecg.` into the ECG submodule;
5. locks/freezes the model;
6. returns `model, preprocess_val, model_config`.

### 6.3 ECG-text data loaders (`ecg_coca/training/data.py`)

The ECG-CoCa training loaders read WFDB ECG records and pair them with text reports or labels.

Key implementation details:

- `ECGTextDataset` loads a WFDB record with `wfdb.rdsamp()`, replaces NaN/Inf with zero, transposes to channel-first shape, optionally applies transforms, and tokenizes lowercased text.
- Dataset helpers exist for PTB-XL reports/diagnostics, CPSC 2018, Chapman-Shaoxing, SPH, and MIMIC-IV-ECG.
- `get_wave_info()` constructs textual descriptions from beat/interval measurements such as RR, PR, QRS, QT/QTc, and P/R/T peaks.
- `get_data()` builds train/validation loaders based on the configured dataset choices.

### 6.4 ECG-CoCa training/evaluation utilities

- `ecg_coca/training/train.py`: epoch training/evaluation/test loops, generative-loss handling, CLIP retrieval metrics, and zero-shot style metrics.
- `ecg_coca/training/evaluation/`: linear probe, zero-shot, metrics, and metadata utilities.
- `scheduler.py`, `distributed.py`, `precision.py`, `logger.py`, `params.py`, `profiler.py`: standard OpenCLIP-style training infrastructure.

## 7. ECG image generation and instruction prompts (`gem_generation/`)

### 7.1 ECG image generator (`gem_generation/ecg-image-generator/`)

This subdirectory is a self-contained ECG paper image synthesis tool. Its README describes it as generating realistic ECG images from WFDB time-series data, including paper-like backgrounds and artifacts such as printing/scanning distortions, handwritten notes, wrinkles, creases, and perspective transforms.

Key components:

- `gen_ecg_images_from_data_batch.py`: batch CLI entry point for generating ECG images from an input directory of WFDB records.
- `gen_ecg_images_from_data_multi.py`: multiprocessing/multi-file generation variant.
- `gen_ecg_image_from_data.py`: single-file generation flow.
- `extract_leads.py`: main high-level function `get_paper_ecg()` that loads record/header data, plots ECGs, applies layout and optional noise/distortions, and emits images plus metadata.
- `ecg_plot.py`: Matplotlib plotting of ECG leads, grid lines, lead names, layout, full lead strip, etc.
- `helper_functions.py`: WFDB/header loading, lead standardization, coordinate conversions, bounding-box utilities, and WFDB output writing.
- `ImageAugmentation/augment.py`: paper scan/image augmentations.
- `TemplateFiles/` and `Fonts/`: text templates and font assets for paper-like output.
- `config.yaml`, `template1.json`, `template2.json`: default generation and paper templates.

Outputs include generated ECG images, modified WFDB header/data files, and optional JSON config/annotation metadata. The tool supports distortionless mode and many flags such as resolution, seed, grid, header printing, QR code, column count, full rhythm strip, random resolution, lead-name removal, and masking unplotted samples.

### 7.2 Instruction-generation prompt (`gem_generation/prompts_generation.txt`)

This prompt template is used to produce grounded ECG analysis text from a ground-truth report plus machine measurements. It instructs the generator to:

- analyze rhythm/rate, intervals, P/QRS/T morphology, ST segments, axis, hypertrophy, bundle branch blocks, infarction/ischemia, and lead-specific abnormalities;
- trust the source report over potentially conflicting machine measurements;
- explicitly ground diagnoses in ECG evidence such as leads, intervals, and beat positions;
- avoid revealing that the report exists;
- output a single paragraph under `**Response:**` limited to about 300 words.

## 8. Evaluation systems (`evaluation/` and `gem_evaluation/`)

### 8.1 GEM benchmark generation scripts (`evaluation/gem_bench/`)

- `bench_ecgbench.sh`: runs `llava/eval/model_ecg_resume.py` for a selected model and dataset split, writing JSONL answers under `eval_outputs`.
- `bench_ecggrounding.sh`: similar generation path for ECG-Grounding; the README notes it is designed for multi-GPU chunked inference.

Both scripts require local editing of model paths, image folders, ECG folders, question files, ECG tower checkpoint path, and `open_clip_config`.

### 8.2 ECG inference scripts (`llava/eval/model_ecg*.py`)

`llava/eval/model_ecg.py` and related variants load a pretrained GEM/LLaVA model with `load_pretrained_model()`, read question JSON/JSONL, construct a prompt using a conversation template, tokenize the `<image>` placeholder, preprocess ECG images, and call `model.generate()`.

`model_ecg_resume.py`/distributed/arena variants add ECG waveform loading and resumable/distributed behavior. The shell benchmark scripts call `model_ecg_resume.py` with `--ecg-folder`, `--ecg_tower`, and `--open_clip_config`, which are required to activate the ECG time-series branch.

### 8.3 ECG-Bench metric evaluation (`evaluation/evaluate_ecgbench.py`)

This script computes task-specific metrics over generated outputs. It contains separate evaluators for multiple ECG-Bench subsets:

- MMMU-like multiple-choice questions.
- PTB-XL diagnostic/report tasks.
- CPSC 2018.
- ECG-QA.
- CODE-15.
- CSN.
- G12E.

Implementation details:

- `extract()` normalizes answers from generated text.
- `compute_f1_auc()` computes F1 and AUC-style metrics using scikit-learn/numpy.
- Each `eval_*` function loads outputs from a directory, parses predictions, compares to target labels, and prints dataset-specific accuracy/F1/AUC-style summaries.

### 8.4 GPT report evaluation (`evaluation/eval_report.py`)

This script evaluates generated reports against golden reports using a GPT judge prompt from `evaluation/prompts.py`.

Flow:

1. Load generated JSONL outputs.
2. Extract or normalize generated text.
3. Build a prompt containing the ground-truth report and generated report.
4. Call an OpenAI/Azure OpenAI chat model for JSON-formatted report scoring.
5. Save one JSON score file per ECG.
6. Aggregate category scores and average report scores.

The script includes cleanup logic for model responses wrapped in markdown code fences.

### 8.5 Grounding evaluation (`gem_evaluation/`)

- `generate_gpt_eval.py` constructs prompts from `prompts_evaluation.txt`, sends generated and ground-truth outputs to an OpenAI model, and saves per-ECG judge results.
- `process_gem_outputs.ipynb` and `process_grounding_scores.ipynb` are notebooks for transforming raw outputs and aggregating grounding scores.
- `prompts_evaluation.txt` defines the GPT judge rubric for comparing GEM-generated explanations against GPT-4o/ground-truth style outputs.

### 8.6 Baseline evaluation (`evaluation/baseline/`)

The baseline folder provides a configurable inference framework for non-GEM or comparison VLMs.

- `config/config.yaml` and `config/prompt/no-refuse.yaml`: model/data/prompt configuration.
- `infer/infer.py`: main baseline inference entry point.
- `infer/models/openai_api.py`: OpenAI API model wrapper.
- `infer/models/qwen2_vl_chat.py`: Qwen2-VL chat wrapper.
- `infer/data_loader.py`, `config_wrapper.py`: data/config abstractions.
- `utils/`: JSONL validation/repair, HF chat templates, image/video-language utilities.

## 9. Serving/deployment (`llava/serve/` and `predict.py`)

The repository retains LLaVA serving utilities:

- `llava/serve/controller.py`, `model_worker.py`, `sglang_worker.py`, `register_worker.py`: controller/worker architecture for hosted models.
- `llava/serve/gradio_web_server.py`: Gradio web UI.
- `llava/serve/cli.py`: command-line chat/inference interface.

`predict.py` is a Cog predictor wrapper. It handles model asset downloads, initializes the model for inference, loads images, and exposes a prediction method suitable for containerized deployment. This file should be compared with upstream if deployment behavior matters, because it can contain path/model assumptions distinct from the main benchmark scripts.

## 10. End-to-end data/model flow

### 10.1 Training example flow

1. Training JSON example contains an ECG waveform path, ECG image path, and conversation.
2. `LazySupervisedDataset` loads waveform from WFDB, normalizes invalid values, transposes to `channels x samples`, pads/truncates to `12 x 5000`, and loads/preprocesses the image.
3. Conversation preprocessing inserts/retains the `<image>` token and masks user turns.
4. The data collator pads token sequences and stacks ECG/image tensors.
5. The LM wrapper receives `input_ids`, `ecgs`, `images`, and `image_sizes`.
6. `prepare_inputs_labels_for_multimodal()` encodes waveform and image, concatenates their token features, replaces the `<image>` token with these features, masks feature positions in labels, and feeds the result as `inputs_embeds` to the LM.
7. Training loss is computed only over unmasked assistant response tokens.

### 10.2 Inference flow

1. Evaluation script loads model/tokenizer/image processor with `load_pretrained_model()`.
2. Script reads question/instruction data and constructs a conversation prompt with `<image>`.
3. It loads/preprocesses image and, in ECG-enabled variants, waveform data.
4. `model.generate()` calls `prepare_inputs_labels_for_multimodal()` to replace `<image>` with concatenated ECG+image embeddings.
5. Hugging Face generation decodes the assistant response.
6. Output is saved as JSONL with question ID, prompt, generated text, model ID, answer ID, and metadata.

## 11. Notable implementation details and comparison targets

These are high-value areas to inspect when comparing with another repository:

1. **ECG fusion uses the image token.** There is no separate `<ecg>` token. ECG and image features are concatenated and inserted at `<image>`.
2. **ECG path has two projectors.** ECG tower features go through `ecg_projector` and then `mm_projector`; image features go only through `mm_projector`.
3. **ECG projector input width is hardcoded to 768.** If the ECG-CoCa config changes width, projector code must change or config must match.
4. **ECG tower loading requires a local checkpoint path.** `build_ecg_tower()` only accepts an existing path and raises otherwise.
5. **The ECG tower is frozen by default.** `CLIPECGTower.load_model()` calls `requires_grad_(False)` and `get_ecg_encoder()` locks the ECG encoder.
6. **Training multimodal activation is tied to `vision_tower`.** The main training branch initializes image processing and sets `is_multimodal` under the `vision_tower is not None` branch; GEM expects both image and ECG tower configuration for full multimodal operation.
7. **Waveform shape assumption is `12 x 5000`.** Both training and ECG tower metadata assume 12-lead ECGs and default 5000 samples.
8. **Any-resolution image logic is inherited from LLaVA.** It supports spatial patch merge and unpadding but requires image size metadata and config pinpoints.
9. **Evaluation scripts contain placeholder paths.** Several scripts must be locally edited before use; this is not a plug-and-play CLI package.
10. **OpenAI evaluation scripts require manual API/model configuration.** API keys and input paths are blank/placeholders in code.
11. **Repository contains large inherited LLaVA functionality.** Many scripts and serving/eval modules are generic LLaVA and may not be GEM-specific.
12. **Potential brittle code spots.** The model builder's Qwen/MoE path and some eval scripts contain hardcoded/default variables and should be tested after path/model changes.

## 12. Suggested comparison checklist

When comparing this repository to another, inspect:

- Whether the other repo has the same `llava_arch.py` ECG branch and whether ECG is inserted via `<image>` or a separate token.
- Whether ECG features pass through one or two projectors.
- The ECG projector dimensions and projector type support.
- How `mm_ecg_tower`, `open_clip_config`, and checkpoint paths are stored in model configs.
- Whether ECG/vision towers are frozen or trainable.
- Dataset schema expected by training (`image`, `ecg`, `conversations`).
- Padding/truncation length for ECG waveforms.
- Conversation template and tokenizer behavior, especially Qwen/Vicuna differences.
- How LoRA/non-LoRA trainables and projector-only checkpoints are saved.
- Which eval scripts are active and whether generated outputs include ECG waveform conditioning.
- Whether GPT judge prompts and score aggregation logic match.
- Whether data-generation prompts or ECG image synthesis settings differ.

## 13. Quick file map by functionality

### Model and fusion

- `llava/model/llava_arch.py` — multimodal ECG/image feature extraction and embedding insertion.
- `llava/model/multimodal_encoder/builder.py` — ECG/vision tower construction.
- `llava/model/multimodal_encoder/clip_encoder.py` — ECG-CoCa and CLIP tower wrappers.
- `llava/model/multimodal_projector/builder.py` — ECG and vision projector builders.
- `llava/model/language_model/*.py` — LM-specific GEM/LLaVA wrappers.
- `llava/model/builder.py` — pretrained model loading for inference/serving.

### Training

- `llava/train/train.py` — SFT entry point, dataset, collator, tokenizer/conversation preprocessing, LoRA/quantization logic.
- `llava/train/llava_trainer.py` — trainer customizations.
- `scripts/train_gem.sh` — GEM training command template.
- `scripts/zero2.json` — DeepSpeed ZeRO-2 config.

### ECG encoder

- `ecg_coca/open_clip/coca_model.py` — CoCa model with ECG/text towers and decoder.
- `ecg_coca/open_clip/model.py` — ECG/text model tower definitions.
- `ecg_coca/training/main.py` — `get_ecg_encoder()` checkpoint bridge.
- `ecg_coca/training/data.py` — ECG-text dataset loaders.
- `ecg_coca/training/train.py` — ECG-CoCa training/eval loops.

### Data generation

- `gem_generation/ecg-image-generator/` — ECG paper image generation package.
- `gem_generation/prompts_generation.txt` — grounded data-generation prompt template.

### Evaluation

- `evaluation/gem_bench/*.sh` — benchmark generation launchers.
- `llava/eval/model_ecg*.py` — GEM/ECG inference scripts.
- `evaluation/evaluate_ecgbench.py` — ECG-Bench metrics.
- `evaluation/eval_report.py` — GPT report evaluation.
- `gem_evaluation/generate_gpt_eval.py` — GPT grounding evaluation.
- `gem_evaluation/*.ipynb` — result post-processing notebooks.
- `evaluation/baseline/` — baseline VLM inference framework.

### Serving/deployment

- `llava/serve/` — LLaVA controller/worker/Gradio/CLI serving stack.
- `predict.py`, `cog.yaml` — Cog-style deployment wrapper.
