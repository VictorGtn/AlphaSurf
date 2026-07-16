# ProteinGym benchmark for AlphaSurf

Zero-shot protein fitness prediction on the ProteinGym substitutions benchmark
(217 DMS assays, single and multi-site mutants), following the reference
implementation of S3F (NeurIPS 2024).

The `option_f` evaluator loads an AlphaSurf S3F-pretraining checkpoint and
follows S3F's released ProteinGym scoring protocol: one masked prediction per
unique mutation-site set, 1,022-residue windows for long sequences, and
mutant-versus-WT log-odds scoring. Because AlphaSurf uses full-atom geometry,
the graph and Alpha-complex surface are regenerated after removing C-beta and
side-chain atoms at each unique masked-position set, matching pretraining.

## Run on Jean-Zay

From the AlphaSurf package directory, submit:

```bash
CKPT=/absolute/path/to/checkpoints/your.ckpt \
sbatch alphasurf/tasks/proteingym/evaluate_jz_h100.sh
```

Use `LIMIT=2` for a first smoke test and optionally set `PROTEINGYM_DIR`,
`OUTPUT_DIR`, or `BATCH_SIZE`. The expected data root contains
`substitutions/DMS_ProteinGym_substitutions/` and
`af2_structures/ProteinGym_AF2_structures/`.

---

## Background

### What ProteinGym measures

For each of 217 DMS assays:
- Input: a wild-type sequence, a list of mutants (e.g. `A123V` or multi-site),
  and an experimentally measured fitness value per mutant.
- Output: a model-predicted score per mutant.
- Metric: Spearman correlation between predicted scores and measured fitness,
  reported per assay. Aggregate stats (mean / NDCG / etc.) are reported across
  all 217 assays.

### How S3F scores a mutant

Reference: `script/evaluate.py` in the S3F repo.

For each DMS assay:
1. Build a WT protein object from the WT sequence, load the AF2 structure, load
   the precomputed surface graph.
2. Load the assay CSV. Each row is `mutant, mutated_sequence, DMS_score`.
3. Mask every mutated position in the input (replace its residue-type feature
   with a `<mask>` token), run the model forward, get logits of shape
   `(seq_len, vocab_size)` over amino-acid types.
4. For each mutant row, sum the position-wise log-odds:
   ```
   logp = log_softmax(logits, dim=-1)
   score = sum over mutated positions i of (logp[i, MT_idx] - logp[i, WT_idx])
   ```
5. Long sequences (>1022 residues, ESM-2 context limit) are handled by
   `get_optimal_window`, which slides a window to cover all mutated positions.
6. Spearman / Pearson / MAE / RMSE between predicted scores and DMS targets.

### S3F model branches

S3F has three input branches: sequence (ESM-2-650M), structure (GearNet on CA
atoms), surface (MaSIF-style graph). The structure and surface branches natively
output `(seq_len, d)` embeddings, not AA-type logits. S3F still scores them with
the log-odds formula above by adding a `Linear(d, vocab_size)` head and training
it via distillation from ESM-2's MLM softmax (paper Section 3.2). At inference
time, ESM-2 is not needed for the structure-only or surface-only models.

---

## Architectural mismatch with AlphaSurf

AlphaSurf's `ProteinEncoder` (in `alphasurf/networks/protein_encoder.py`)
outputs continuous per-residue graph embeddings and per-vertex surface
embeddings of shape `(N, d)`. There is **no residue-type prediction head**.
The trained checkpoint at `tasks/pinder_pair/ckpt/last.ckpt` was trained on the
PINDER binary contact-prediction task; its heads predict interface contacts and
binding-site probabilities, not AA identity.

The log-odds scoring formula needs a categorical distribution over AA types at
each position. That requires one of the options below.

---

## Options for closing the gap

### Option A: Distill ESM-2 into a head on top of the AlphaSurf encoder

This is what S3F did for its structure-only and surface-only models. ESM-2 is
used only as a teacher during the head-training stage; it is not needed at
inference.

Procedure:
1. Take a corpus of UniRef or PDB sequences (a few hundred thousand is enough).
2. For each sequence, run ESM-2 once and cache the per-position softmax over
   AA types (`(seq_len, 33)` including special tokens, or `(seq_len, 20)` for
   the standard AA alphabet).
3. Run AlphaSurf's encoder on the same sequence's structure (AF2 predicted if
   no experimental structure is available).
4. Train a `Linear(d, 20)` head (and optionally LoRA / last-few-layers of the
   encoder) so that `log_softmax(head(graph_emb))` matches ESM-2's softmax via
   KL divergence. Mask a random 15% of input residue types so the encoder is
   forced to use context rather than the residue identity itself.
5. At inference, score mutants with the standard log-odds formula.

Cost: ~1 GPU-day for the distillation pass on a corpus of ~100k proteins.
Output: numbers directly comparable to S3F's s2f (structure-only) and s3f
(structure + surface) rows in the ProteinGym table.

### Option B: Train AlphaSurf from scratch as an MLM

Skip distillation; train the whole model with a random-mask objective over a
large sequence corpus. This is the cleanest comparison to ESM-2 / S3F's
sequence branch and is the standard setup for ProteinGym baselines.

Cost: many GPU-weeks. Output: numbers fully comparable to the top of the
ProteinGym leaderboard.

### Option C: Regression head on DMS or DDG data

Train `Linear(d, 1)` to predict fitness or DDG from the WT embedding plus
mutation encoding. Not strictly zero-shot (depends on the training set) and
uses a different reference set than the standard ProteinGym substitutions
benchmark. Useful as a complementary supervised number.

### Option D: Embedding-delta heuristic (no training)

Encode WT, encode MT (introduce the mutation in the input residue features),
take `score = -||graph_emb_MT(i) - graph_emb_WT(i)||` at the mutated position.
For multi-site mutants, sum over positions.

Pros: zero training, runs immediately on the released PINDER checkpoint.
Cons: non-standard. Not a log-odds, so the Spearman number is not comparable
to anything in the ProteinGym leaderboard. Measures embedding locality under
perturbation, which is correlated with fitness but not the same quantity.
Reviewers will rightly push back on reporting this as a ProteinGym result.

Useful only as a quick sanity check that the encoder carries useful
mutation-sensitive signal at all.

---

## Recommended path

Option A is the recommended primary target: it is what S3F did for its
structure / surface branches, it produces numbers comparable to the S3F paper,
and it exercises AlphaSurf's surface encoder rather than just calling ESM-2.
Option D is a useful same-day sanity check before committing to the
distillation pass.

Open questions to resolve before coding:
- Which encoder checkpoint to distill from? The released PINDER checkpoint is
  fine but not directly comparable to S3F (which was trained on UniRef). For a
  fair comparison, an AlphaSurf encoder pretrained on a comparable MLM or
  UniRef objective is preferable.
- Which AA alphabet: 20 standard, or ESM-2's full 33-token alphabet? Use 20
  for comparability with S3F's reported numbers; the 4 non-standard residues
  in ProteinGym are folded into `X`.
- Distillation corpus: UniRef50 is standard. PDB chains are also reasonable
  but smaller.

---

## Data

Download (mirror S3F; skip the surface zip because AlphaSurf generates surfaces
on the fly via the CGAL alpha-complex bindings):

- `DMS_ProteinGym_substitutions.zip` from `marks.hms.harvard.edu` - 217 assay
  CSVs, columns `mutant, mutated_sequence, DMS_score`.
- `ProteinGym_AF2_structures.zip` - one AF2 PDB per UniProt ID, used as the
  input structure for each assay.

Each assay CSV maps to one UniProt ID. The mapping is implicit in the filename
and in the CSV's `UniProt_ID` column. The WT sequence is the un-mutated
reference; the PDB structure for the assay is
`ProteinGym_AF2_structures/{UniProt_ID}.pdb`.

---

## File layout (planned)

Mirror the structure of `tasks/pinder_pair/`:

```
tasks/proteingym/
    README.md                   this file
    __init__.py
    datamodule.py               LightningDataModule: iterates the 217 assays
    dataset.py                  per-assay dataset: WT sequence, mutants, targets
    model.py                    AlphaSurf encoder + Linear(d, 20) head
    pl_model.py                 AtomPLModule subclass (train + zero-shot modes)
    scoring.py                  log-odds scoring + optimal window for long seqs
    evaluate.py                 CLI: --datadir, --structdir, --ckpt
    distill.py                  CLI: distill ESM-2 softmax into AlphaSurf head
    conf/
        evaluate_zeroshot.yaml
        distill.yaml
    download_proteingym.sh      fetches DMS csvs + AF2 structures
```

### Reused infrastructure

- `alphasurf/protein/protein_loader.py::ProteinLoader` - builds graph + surface
  from a PDB on the fly via the alpha-complex bindings. Replaces S3F's
  precomputed MaSIF surface step entirely.
- `alphasurf/protein/create_esm.py` - already wraps `esm.pretrained.esm2_t33_650M_UR50D()`,
  the same model S3F uses. Reuse for distillation teacher.
- `alphasurf/protein/graphs.py` - `res_type_dict` (20 AA + UNK), `protein_letters_1to3`,
  `quick_pdb_to_seq`. Use the existing UNK index (20) as the `<mask>` token.
- `alphasurf/utils/learning_utils.py::AtomPLModule` - base class for `pl_model.py`.
- `alphasurf/tasks/pinder_pair/model.py::PinderPairNet` - template for
  `model.py`: how to wrap `ProteinEncoder` with task-specific heads.
- `alphasurf/tasks/inference/embed.py` - template for the per-protein forward
  pass + batch pointer slicing.

### Code to port verbatim from S3F

These helpers in S3F's `script/evaluate.py` are model-agnostic given a
`(seq_len, 20)` logit tensor:
- `evaluate(pred, target)` - spearman / pearson / MAE / RMSE.
- `get_optimal_window(mut_pos, seq_len, model_window)` - long-sequence windowing.
- `get_prob(seq_prob, mutations, offsets)` - per-mutant log-odds extraction,
  with minor adaptation to AlphaSurf's residue indexing.
- The mutant-position masking logic in `load_dataset`.

### Code that needs to be written fresh

- `model.py::ProteinGymNet` - `ProteinEncoder` + `Linear(d, 20)` head. The head
  reads the per-residue graph embedding and outputs AA-type logits. The encoder
  runs unchanged.
- `pl_model.py::ProteinGymModule` - the training step computes KL divergence
  between `log_softmax(head(graph_emb))` and cached ESM-2 softmaxes (distillation
  mode), or cross-entropy against the true AA identity (from-scratch MLM mode).
- `scoring.py::score_mutants(graph_emb, head, mutants)` - the log-odds loop.
- `evaluate.py` - per-assay loop over the 217 CSVs; loads structure via
  `ProteinLoader`, builds the mutant dataset, runs inference, writes per-assay
  CSVs with model scores and a top-level summary CSV with one row per assay.
- `distill.py` - the distillation training loop: enumerates a sequence corpus,
  runs ESM-2 once per sequence to cache softmaxes, then trains the head.

### Special cases in S3F's evaluate.py

Three assays have hardcoded residue-index ranges in S3F's code (POLG_HCVJF_Qi_2014,
A0A140D2T1_ZIKV_Sourisseau_2019, B2L11_HUMAN_Dutta_2010_binding-Mcl-1) because
the assay's residue numbering does not match the UniProt sequence numbering.
These offsets need to be preserved when porting `load_dataset`. They are a
property of the ProteinGym data, not of the model.

---

## Outputs

Per-assay CSV (one per DMS assay):
```
,mutant,mutated_sequence,DMS_score,model_score
```

Summary CSV (one row per assay):
```
DMS_id,UniProt_ID,seq_len,DMS_number_single_mutants,spearmanr,pearsonr,mae,rmse
```

Aggregate stats across all 217 assays: mean spearman, median spearman, std
spearman, mean NDCG. These are the numbers to report.

---

## Reference links

- S3F repo: https://github.com/DeepGraphLearning/S3F
- S3F paper: "Multi-Scale Representation Learning for Protein Fitness Prediction"
  (NeurIPS 2024).
- ProteinGym: https://proteingym.org
- DMS data: `marks.hms.harvard.edu` (S3F README has exact URLs).
- ESM-2 checkpoint: `dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt`
  (same model the existing `alphasurf/protein/create_esm.py` loads).
