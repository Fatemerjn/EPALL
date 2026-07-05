# Graph Report - .  (2026-07-05)

## Corpus Check
- 103 files · ~81,608 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1044 nodes · 2294 edges · 60 communities (51 shown, 9 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 186 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Subnet Mask Layers|Subnet Mask Layers]]
- [[_COMMUNITY_Training Main Metrics|Training Main Metrics]]
- [[_COMMUNITY_Swift Overlap Plots|Swift Overlap Plots]]
- [[_COMMUNITY_Dataset Loading|Dataset Loading]]
- [[_COMMUNITY_PALL Adapter Logic|PALL Adapter Logic]]
- [[_COMMUNITY_Device Overlap Analysis|Device Overlap Analysis]]
- [[_COMMUNITY_Method Review Docs|Method Review Docs]]
- [[_COMMUNITY_PALL Base Forgetting|PALL Base Forgetting]]
- [[_COMMUNITY_Paper Figures|Paper Figures]]
- [[_COMMUNITY_Thesis Table Builder|Thesis Table Builder]]
- [[_COMMUNITY_Overlap Summaries|Overlap Summaries]]
- [[_COMMUNITY_Base Model Wrapper|Base Model Wrapper]]
- [[_COMMUNITY_Report Plotting|Report Plotting]]
- [[_COMMUNITY_Regularization Methods|Regularization Methods]]
- [[_COMMUNITY_Experiment Summaries|Experiment Summaries]]
- [[_COMMUNITY_Replay Methods|Replay Methods]]
- [[_COMMUNITY_Graphify Instruction Docs|Graphify Instruction Docs]]
- [[_COMMUNITY_Adapter Ablation Summary|Adapter Ablation Summary]]
- [[_COMMUNITY_Results Aggregation|Results Aggregation]]
- [[_COMMUNITY_Overlap Analysis|Overlap Analysis]]
- [[_COMMUNITY_Paper Summary Table|Paper Summary Table]]
- [[_COMMUNITY_Report Table Builder|Report Table Builder]]
- [[_COMMUNITY_Method Modules|Method Modules]]
- [[_COMMUNITY_Model Factories|Model Factories]]
- [[_COMMUNITY_Lifelong Baselines|Lifelong Baselines]]
- [[_COMMUNITY_LoRA ResNet|LoRA ResNet]]
- [[_COMMUNITY_Project Cleanup|Project Cleanup]]
- [[_COMMUNITY_Ablation Table|Ablation Table]]
- [[_COMMUNITY_Candidate Overlap Search|Candidate Overlap Search]]
- [[_COMMUNITY_Adapter ResNet|Adapter ResNet]]
- [[_COMMUNITY_Adapter Ablation Runner|Adapter Ablation Runner]]
- [[_COMMUNITY_LoRA Method Logic|LoRA Method Logic]]
- [[_COMMUNITY_LoRA Modules|LoRA Modules]]
- [[_COMMUNITY_ResNet Backbone|ResNet Backbone]]
- [[_COMMUNITY_Markdown Latex Tables|Markdown Latex Tables]]
- [[_COMMUNITY_Thesis Results Plotting|Thesis Results Plotting]]
- [[_COMMUNITY_Controlled Experiments|Controlled Experiments]]
- [[_COMMUNITY_Schedule Generation|Schedule Generation]]
- [[_COMMUNITY_Comparison Tables|Comparison Tables]]
- [[_COMMUNITY_PALL Ablation Runner|PALL Ablation Runner]]
- [[_COMMUNITY_Overlap Schedule Search|Overlap Schedule Search]]
- [[_COMMUNITY_Forgetting Regression Plot|Forgetting Regression Plot]]
- [[_COMMUNITY_Server Experiment Script|Server Experiment Script]]
- [[_COMMUNITY_CLPU Method|CLPU Method]]
- [[_COMMUNITY_Baseline Runner|Baseline Runner]]
- [[_COMMUNITY_Fixed Schedule Baselines|Fixed Schedule Baselines]]
- [[_COMMUNITY_Bottleneck Adapters|Bottleneck Adapters]]
- [[_COMMUNITY_Base Model Params|Base Model Params]]
- [[_COMMUNITY_Pretrained Backbone|Pretrained Backbone]]
- [[_COMMUNITY_PALL Pairwise Runner|PALL Pairwise Runner]]
- [[_COMMUNITY_Small Ablation Runner|Small Ablation Runner]]
- [[_COMMUNITY_Request Schedule Export|Request Schedule Export]]
- [[_COMMUNITY_Overlap Metrics|Overlap Metrics]]
- [[_COMMUNITY_Thesis Invocation Image|Thesis Invocation Image]]
- [[_COMMUNITY_Paper Experiment Script|Paper Experiment Script]]
- [[_COMMUNITY_Graphify Root Instructions|Graphify Root Instructions]]
- [[_COMMUNITY_Smoke Test Script|Smoke Test Script]]
- [[_COMMUNITY_Example Run Script|Example Run Script]]
- [[_COMMUNITY_Dataset Constraints|Dataset Constraints]]

## God Nodes (most connected - your core abstractions)
1. `PALLBase` - 40 edges
2. `SubnetVisionTransformer` - 38 edges
3. `PALLAdapter` - 37 edges
4. `main()` - 32 edges
5. `Base` - 32 edges
6. `VisionTransformer` - 30 edges
7. `SubnetLinear` - 26 edges
8. `SubnetConv2d` - 26 edges
9. `LoRAResNet` - 23 edges
10. `AdapterResNet` - 22 edges

## Surprising Connections (you probably didn't know these)
- `Method Taxonomy` --semantically_similar_to--> `Implemented Methods`  [INFERRED] [semantically similar]
  paper/REVIEW_NOTES.md → README.md
- `Phase-3 Iterative Uniform-Target Loop` --semantically_similar_to--> `PALL-Adapter`  [INFERRED] [semantically similar]
  paper/REVIEW_NOTES.md → README.md
- `Graphify Instructions` --semantically_similar_to--> `Graphify Instructions`  [INFERRED] [semantically similar]
  AGENTS.md → CLAUDE.md
- `Gradient-Magnitude Importance Resolution` --semantically_similar_to--> `PALL-Modified Gradient Importance`  [INFERRED] [semantically similar]
  paper/REVIEW_NOTES.md → README.md
- `Gradient-Conflict Protection` --semantically_similar_to--> `PALL-Modified Conflict`  [INFERRED] [semantically similar]
  paper/REVIEW_NOTES.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Graphify Navigation Contract** — agents_graphify, claude_graphify, agents_graphify_query_path_explain [INFERRED 0.95]
- **PALL Method Taxonomy** — readme_pall_original, readme_pall_modified, readme_pall_adapter, readme_baseline_methods, paper_review_notes_method_taxonomy [INFERRED 0.95]
- **Overlap Protection Mechanisms** — readme_overlap_aware_forgetting, readme_pall_modified_conflict, readme_s_share_crit, paper_review_notes_gradient_conflict_protection, paper_review_notes_conflict_energy, paper_review_notes_phase3_iterative_uniform_target_loop [INFERRED 0.85]
- **Besmellah Calligraphic Invocation** — thesis_front_template_images_besmellah_image, thesis_front_template_images_besmellah_bismillah_al_rahman_al_rahim, thesis_front_template_images_besmellah_arabic_calligraphy, thesis_front_template_images_besmellah_islamic_invocation [INFERRED 0.85]
- **Graphify Instruction Runbooks** — claude_claude_claude_graphify_claude_instructions, claude_skills_graphify_skill_graphify_skill_pipeline, claude_skills_graphify_references_add_watch_graphify_add_watch_flow, claude_skills_graphify_references_exports_graphify_export_targets, claude_skills_graphify_references_extraction_spec_graphify_extraction_schema, claude_skills_graphify_references_github_and_merge_graphify_github_merge_flow, claude_skills_graphify_references_hooks_graphify_hook_integration, claude_skills_graphify_references_query_graphify_query_traversal, claude_skills_graphify_references_transcribe_graphify_transcription_flow, claude_skills_graphify_references_update_graphify_update_flow, codex_skills_graphify_skill_graphify_skill_pipeline, codex_skills_graphify_references_add_watch_graphify_add_watch_flow [INFERRED 0.85]

## Communities (60 total, 9 thin omitted)

### Community 0 - "Subnet Mask Layers"
Cohesion: 0.07
Nodes (22): MaskByScores, SubnetClassifier, SubnetConv2d, SubnetLinear, maskedSequential, subnet_conv1x1(), subnet_conv3x3(), subnet_resnet18() (+14 more)

### Community 1 - "Training Main Metrics"
Cohesion: 0.08
Nodes (51): acc_list_to_dict(), avg_for_tasks(), build_requests_with_active_tasks(), _coerce_float_or_none(), _coerce_int_or_none(), compute_average_forgetting(), compute_mia(), compute_unlearning_score() (+43 more)

### Community 2 - "Swift Overlap Plots"
Cohesion: 0.08
Nodes (52): AppKit, CustomStringConvertible, Error, Foundation, drawMarker(), drawPlot(), drawText(), loadRows() (+44 more)

### Community 3 - "Dataset Loading"
Cohesion: 0.07
Nodes (29): get_cifar100_superclass_tasks(), get_dataset_metadata(), get_task_datasets(), _load_tinyimagenet_wnids(), _resolve_tinyimagenet_root(), SubDataset, TinyImageNetTrainDataset, TinyImageNetValDataset (+21 more)

### Community 4 - "PALL Adapter Logic"
Cohesion: 0.12
Nodes (6): PALLAdapter, Parameter-efficient, overlap-aware adapter forgetting.      Architecture: frozen, Cross-entropy of the forget task's logits against a UNIFORM target.          Imp, Per-parameter gradient-CONFLICT energy on the shared adapter.          ``relu(-g, Single gradient-ASCENT step on the forget task's true-label loss.          The l, Phase 3: ITERATIVE uniform-target soft-masked forgetting on the shared adapter.

### Community 5 - "Device Overlap Analysis"
Cohesion: 0.17
Nodes (35): Select an execution device with macOS MPS support., resolve_device(), RuntimeError, build_correlation_summary(), build_row(), canonicalize_method_variant(), extract_overlap_analysis(), extract_overlap_from_csv() (+27 more)

### Community 6 - "Method Review Docs"
Cohesion: 0.07
Nodes (34): Configs Directory, Code Review Notes and Open Decisions, Conflict Energy, Gradient-Conflict Protection, Gradient-Magnitude Importance Resolution, Method Taxonomy, Phase-3 Iterative Uniform-Target Loop, Requirements Fix (+26 more)

### Community 7 - "PALL Base Forgetting"
Cohesion: 0.12
Nodes (7): PALLBase, Subnet-mask PALL with overlap-aware selective forgetting (shared base).      Thi, Select the *critical* subset of the forget/retain shared parameters.          Th, Fresh, data-independent reinit sample for the masked positions.          Used by, Per-parameter SIGNED gradient of the summed CE loss over the given         tasks, Per-parameter |grad L_retain| over the rehearsal buffer (S_active).          Abs, Per-parameter gradient-CONFLICT energy on the rehearsal buffer.          For eac

### Community 8 - "Paper Figures"
Cohesion: 0.12
Nodes (31): _betacf(), _betai(), _find_col(), _linregress(), _load_metric_by_method(), _load_overlap_points(), main(), _make_localizer() (+23 more)

### Community 9 - "Thesis Table Builder"
Cohesion: 0.21
Nodes (31): aggregate_group(), build_table(), canonicalize_method_variant(), config_group_value(), derive_adapter_param_ratio(), derive_unlearning_score(), derive_updated_param_ratio(), extract_run_row() (+23 more)

### Community 10 - "Overlap Summaries"
Cohesion: 0.22
Nodes (30): aggregate_group(), build_row(), build_summary(), canonicalize_method_variant(), derive_adapter_param_ratio(), derive_updated_param_ratio(), extract_legacy_overlap_counts(), extract_method_variant() (+22 more)

### Community 11 - "Base Model Wrapper"
Cohesion: 0.14
Nodes (4): Base, Deterministically reseed NumPy/Python RNGs inside each DataLoader worker.      P, Lazily build a CPU generator seeded from the run seed.          Passing an expli, _seed_worker()

### Community 12 - "Report Plotting"
Cohesion: 0.24
Nodes (22): Figure, build_forgetting_quality_points(), build_tradeoff_points(), dedup_legend(), group_by_dataset(), main(), make_bar_plot(), make_forgetting_quality_scatter() (+14 more)

### Community 13 - "Regularization Methods"
Cohesion: 0.18
Nodes (8): EWC, Sequential, subnet_vit_t_16(), subnet_vit_t_8(), SubnetVisionTransformer, VisionTransformer, vit_t_16(), vit_t_8()

### Community 14 - "Experiment Summaries"
Cohesion: 0.21
Nodes (20): build_metric_table(), build_observations(), build_overlap_table(), describe_difference(), fmt_mean_std(), fmt_number(), group_means(), group_results() (+12 more)

### Community 15 - "Replay Methods"
Cohesion: 0.16
Nodes (3): Derpp, ER, RehearsalMemory

### Community 16 - "Graphify Instruction Docs"
Cohesion: 0.15
Nodes (19): Graphify Claude Instructions, Graphify Add Watch Flow, Graphify Export Targets, Graphify Extraction Schema, Graphify GitHub Merge Flow, Graphify Hook Integration, Graphify Query Traversal, Graphify Transcription Flow (+11 more)

### Community 17 - "Adapter Ablation Summary"
Cohesion: 0.22
Nodes (18): ablation_name_for_tag(), compact_csv_row(), compact_markdown_row(), dedupe_ablation_settings(), dedupe_rows(), format_mean_std(), format_number(), main() (+10 more)

### Community 18 - "Results Aggregation"
Cohesion: 0.31
Nodes (17): aggregate_runs(), extract_row(), find_run_dirs(), first_non_none(), get_final_unlearning_block(), get_last_raw_unlearning_event(), is_smoke_run(), load_json() (+9 more)

### Community 19 - "Overlap Analysis"
Cohesion: 0.31
Nodes (17): build_row(), extract_ratio_stats(), find_overlap_files(), first_non_none(), fmt(), load_json(), main(), mean_or_none() (+9 more)

### Community 20 - "Paper Summary Table"
Cohesion: 0.25
Nodes (17): choose_best_adapter_rows(), compact_adapter_row(), compact_report_row(), format_mean_std(), format_number(), main(), normalize_report_value(), parse_args() (+9 more)

### Community 21 - "Report Table Builder"
Cohesion: 0.24
Nodes (17): compact_row(), dedupe_key(), dedupe_rows(), first_present_value(), format_mean_std(), format_number(), main(), markdown_cell() (+9 more)

### Community 22 - "Method Modules"
Cohesion: 0.24
Nodes (5): PALLModified, PALL-Modified -- the MAIN overlap-aware selective-forgetting method.  Identifies, PALLOriginal, PALL-Original -- the PALL baseline (no overlap protection).  Forgetting resets t, Backward-compatibility shim.  The PALL implementation was split into:   * ``meth

### Community 23 - "Model Factories"
Cohesion: 0.21
Nodes (9): adapter_resnet18(), adapter_resnet34(), adapter_resnet50(), TaskBottleneckAdapter, lora_resnet18(), lora_resnet34(), lora_resnet50(), BasicBlock (+1 more)

### Community 24 - "Lifelong Baselines"
Cohesion: 0.22
Nodes (4): LSF, LwF, modified_kl_div(), smooth()

### Community 26 - "Project Cleanup"
Cohesion: 0.33
Nodes (13): collect_backup_files(), collect_incomplete_run_dirs(), collect_pyc_files(), collect_pycache_dirs(), collect_root_output_files(), ensure_parent(), main(), move_file() (+5 more)

### Community 27 - "Ablation Table"
Cohesion: 0.32
Nodes (13): build_table(), fmt_mean_std(), has_adapter_ablation(), main(), mean_std(), normalize_group_value(), parse_args(), parse_float() (+5 more)

### Community 28 - "Candidate Overlap Search"
Cohesion: 0.32
Nodes (13): apply_tag_suffix(), build_commands(), DatasetPreset, main(), method_args(), parse_args(), print_command(), protect_ratio_tag() (+5 more)

### Community 30 - "Adapter Ablation Runner"
Cohesion: 0.40
Nodes (12): AblationConfig, build_command(), build_commands(), DatasetPreset, main(), parse_args(), print_command(), Namespace (+4 more)

### Community 32 - "LoRA Modules"
Cohesion: 0.23
Nodes (4): Per-task LoRA module: x + (alpha/r) * B(A x), no nonlinearity.      Mirrors ``Ta, Optional shared LoRA applied to the feature for all tasks., SharedLoRA, TaskLoRA

### Community 33 - "ResNet Backbone"
Cohesion: 0.30
Nodes (6): conv1x1(), conv3x3(), ResNet, resnet18(), resnet34(), resnet50()

### Community 34 - "Markdown Latex Tables"
Cohesion: 0.30
Nodes (11): escape_latex(), extract_table(), is_separator_row(), is_table_line(), main(), parse_args(), Namespace, Path (+3 more)

### Community 35 - "Thesis Results Plotting"
Cohesion: 0.29
Nodes (11): build_legend(), filtered_points(), main(), make_plot(), parse_args(), parse_float(), Any, Axes (+3 more)

### Community 36 - "Controlled Experiments"
Cohesion: 0.39
Nodes (11): build_commands(), DatasetPreset, main(), method_args(), parse_args(), print_command(), Namespace, Path (+3 more)

### Community 37 - "Schedule Generation"
Cohesion: 0.40
Nodes (10): build_payload(), build_requests_from_plan(), load_main_schedule_validator(), main(), parse_args(), Any, Namespace, Path (+2 more)

### Community 38 - "Comparison Tables"
Cohesion: 0.38
Nodes (10): build_table(), fmt_mean_std(), main(), mean_std(), parse_args(), parse_float(), Path, read_rows() (+2 more)

### Community 39 - "PALL Ablation Runner"
Cohesion: 0.40
Nodes (10): build_commands(), main(), parse_args(), parse_bool_token(), parse_retrain_steps(), Namespace, Path, run_commands() (+2 more)

### Community 40 - "Overlap Schedule Search"
Cohesion: 0.42
Nodes (10): build_payload(), build_requests(), load_main_schedule_validator(), main(), parse_args(), Any, Namespace, Path (+2 more)

### Community 41 - "Forgetting Regression Plot"
Cohesion: 0.33
Nodes (9): DataFrame, ndarray, compute_regression_band(), main(), parse_args(), plot_publication_figure(), Namespace, Path (+1 more)

### Community 42 - "Server Experiment Script"
Cohesion: 0.51
Nodes (9): group1(), group2(), group3(), group4(), group5(), group6_standard(), group7_tiny(), launch() (+1 more)

### Community 44 - "Baseline Runner"
Cohesion: 0.53
Nodes (8): build_commands(), main(), parse_args(), Namespace, Path, run_commands(), validate_args(), write_commands_log()

### Community 45 - "Fixed Schedule Baselines"
Cohesion: 0.56
Nodes (8): build_commands(), main(), parse_args(), Namespace, Path, run_commands(), validate_args(), write_commands_log()

### Community 48 - "Pretrained Backbone"
Cohesion: 0.29
Nodes (4): build_frozen_backbone(), FrozenImageNetBackbone, Return a FrozenImageNetBackbone for the named option, or None for 'none'., ImageNet-pretrained ResNet-18 feature extractor: frozen, offline-safe.      Wrap

### Community 49 - "PALL Pairwise Runner"
Cohesion: 0.61
Nodes (7): build_commands(), main(), parse_args(), Namespace, Path, run_commands(), validate_args()

### Community 50 - "Small Ablation Runner"
Cohesion: 0.61
Nodes (7): build_commands(), main(), parse_args(), Namespace, Path, run_commands(), validate_args()

### Community 51 - "Request Schedule Export"
Cohesion: 0.43
Nodes (6): generate_user_requests(), main(), parse_args(), Namespace, Mirror main.py request generation behavior., with_active_tasks()

### Community 52 - "Overlap Metrics"
Cohesion: 0.50
Nodes (4): critical_ratio, overlap_analysis, protected_ratio, S_share_crit

### Community 53 - "Thesis Invocation Image"
Cohesion: 0.67
Nodes (4): Arabic Calligraphy, Bismillah al-Rahman al-Rahim, Besmellah Image, Islamic Invocation

### Community 54 - "Paper Experiment Script"
Cohesion: 0.83
Nodes (3): launch(), run_dataset(), run_paper_experiments.sh script

### Community 55 - "Graphify Root Instructions"
Cohesion: 0.67
Nodes (3): Graphify Instructions, Graphify Query Path Explain, Graphify Instructions

## Ambiguous Edges - Review These
- `results aggregates` → `Configs Directory`  [AMBIGUOUS]
  configs/README.md · relation: conceptually_related_to

## Knowledge Gaps
- **21 isolated node(s):** `example_run.sh script`, `DatasetPreset`, `DatasetPreset`, `Graphify Query Path Explain`, `Graphify Instructions` (+16 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `results aggregates` and `Configs Directory`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `main()` connect `Training Main Metrics` to `PALL Adapter Logic`, `CLPU Method`, `Regularization Methods`, `Replay Methods`, `Method Modules`, `Lifelong Baselines`, `LoRA Method Logic`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `SubnetVisionTransformer` connect `Regularization Methods` to `Subnet Mask Layers`, `PALL Base Forgetting`, `CLPU Method`, `Replay Methods`, `Base Model Params`, `Method Modules`, `Lifelong Baselines`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `PALLAdapter` connect `PALL Adapter Logic` to `Training Main Metrics`, `Base Model Wrapper`, `Method Modules`, `Replay Methods`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `PALLBase` (e.g. with `RehearsalMemory` and `.__init__()`) actually correct?**
  _`PALLBase` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `SubnetVisionTransformer` (e.g. with `CLPU` and `.learn()`) actually correct?**
  _`SubnetVisionTransformer` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `PALLAdapter` (e.g. with `main()` and `RehearsalMemory`) actually correct?**
  _`PALLAdapter` has 3 INFERRED edges - model-reasoned connections that need verification._