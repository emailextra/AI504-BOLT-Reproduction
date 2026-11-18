# BOLT reproduction on BANKING77

## 1 Paper and claim

Chuan Qin, Xi Chen, Jinpeng Li, and Hengshu Zhu. 2026. BOLT: Benchmarking Open-World Learning for Text Classification. Findings of ACL 2026, pp. 13624–13658. https://aclanthology.org/2026.findings-acl.667/

The main reproduction target is DeepUNK on BANKING77 at labeled ratio 0.10, known-class ratios 0.25 and 0.50, and all five official class partitions. Published known-class F1 is 37.49 and 48.99; published unknown-class F1 is 4.16 and 31.21. An additional five-epoch KNNCon study on partition zero tests released versus corrected supervision masks. Both distance variants are reported. See Report.pdf, Tables 1–3.

## 2 Environment

The intended model runtime is Linux, Python 3.11, and an NVIDIA Tesla T4 or equivalent CUDA GPU. Successful evidence used a T4 with 15,360 MiB total GPU memory. Use BOLT_Colab_Final.ipynb for isolated GPU environment installation and actual PyTorch and TensorFlow LSTM checks. Its first cell contains the project scripts, so no input ZIP is needed.

requirements-torch.txt and requirements-tensorflow.txt pin the model dependencies. TensorFlow CUDA libraries are loaded only in its subprocess. The training launcher uses the pinned Python executable explicitly. Recorded package inventories are in results/environment.txt. Successful model runs have been validated in Colab; the final reorganized launcher has syntax and evidence checks, and awaits a fresh model rerun.

For local Linux use, install uv==0.9.22 and follow this setup from the repository root.

```bash
uv venv --python 3.11 .venv-torch
uv venv --python 3.11 .venv-tf
uv pip install --python .venv-torch/bin/python -r requirements-torch.txt
uv pip install --python .venv-tf/bin/python -r requirements-tensorflow.txt
source .venv-torch/bin/activate
export MPLBACKEND=Agg
export TF_USE_LEGACY_KERAS=1
export BOLT_TF_PYTHON="$PWD/.venv-tf/bin/python"
export BOLT_TF_LIBS="$(find "$PWD/.venv-tf/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd: -):${LD_LIBRARY_PATH:-}"
```

The notebook performs equivalent setup and tests actual GPU operations before training. Use a new results directory for independent training. Regenerating the supplied report tables and running the evidence tests require only Python 3.11 and the repository files.

## 3 Data and preprocessing

Run python code/train.py prepare to fetch the authors' repository at commit 2c935438b5506febcfd23e499e3de81868236b17. This creates .bolt/. Use the Python executable from the pinned environment for training commands.

The data source is BOLT/data/banking. It contains origin_data/{train,dev,test}.tsv, labeled_data/0.1/{train,dev}.tsv and label/fold5/part*/label_known_*.list. The stored masks align with original row order. No resplitting is performed. The released model runners tokenize the stored text, so the train commands below are also the exact preprocessing entry points.

BANKING77 source and licence information: https://huggingface.co/datasets/PolyAI/banking77 (CC BY 4.0). BERT checkpoint: https://huggingface.co/google-bert/bert-base-uncased. GloVe: https://nlp.stanford.edu/projects/glove/. The DeepUNK command downloads the approximately 862 MB GloVe archive and extracts its 300-dimensional file. Dataset and model assets are acquired by the scripts. Check upstream terms before redistributing assets.

## 4 Commands for reported results

Run from the repository root. Analysis and evidence checks use Python standard-library modules. Model commands use the pinned GPU environments and the environment configuration provided in the notebook.

| Evidence | Command | Output |
| --- | --- | --- |
| Every table and independent metric audit | python code/analyze_results.py | results/tables/*.csv |
| Reduced model demonstration | python code/train.py quick --output results/rerun | results/rerun/quick/ |
| DeepUNK five-partition reproduction | python code/train.py deepunk --output results/rerun | results/rerun/full/ |
| KNNCon released and corrected five-epoch comparison | python code/train.py paired --output results/rerun | results/rerun/short_5epoch/ |
| Tables for a complete independent rerun | python code/analyze_results.py --results results/rerun | results/rerun/tables/ |

The saved mask audit is generated from the pinned data with python code/author_runner.py audit --upstream .bolt --output results/rerun. Generate this file before analyzing an independent rerun. The combined analysis requires all ten DeepUNK runs, the paired runs, and the mask audit. A reduced demonstration alone produces its own results.csv.

Equivalent bash wrappers are in scripts/. Tables 1, 2, and 3 in the report correspond to deepunk_comparison.csv, knncon_paired.csv, and errors.csv respectively. Failure examples derive from the corresponding prediction rows and failure_examples.csv. Runtime evidence is runtime.csv. The report builder regenerates figures from these tables.

## 5 Expected outputs and measured time

The complete saved evidence is already supplied. Regenerating tables takes less than a second on the validation machine. Ten DeepUNK runs took 19.97 GPU minutes in total. The paired KNNCon runs took 620.76 and 100.80 seconds. The reduced run took 176.59 seconds. Downloads, installation, and framework startup vary by connection and hardware.

DeepUNK means (known F1, unknown F1) are (36.423, 1.188) at ratio 0.25 and (47.773, 32.995) at ratio 0.50. Released cosine KNNCon gives (69.80, 81.59); corrected cosine gives (11.60, 14.34). Predictions and metric CSVs are verified within rounding tolerance by the analysis command.

For the instructor's rerun tolerance, compare a rerun against our report within one percentage point or the relevant reported standard deviation, whichever is larger. The published-versus-reproduced comparison is a separate assessment. Fold standard deviations measure class-partition variation. The paired KNNCon experiment has one seed and one partition.

A trial is complete only when run.json exists. The launcher skips those trials in the selected output directory. An interrupted trial restarts. Independent reruns use results/rerun to keep supplied evidence available. If importing a prior evidence ZIP into Colab, restore its paths under results/ before training.

## 6 Reused and original code

code/author_runner.py stages training code from the pinned BOLT checkout. Training algorithms, released preprocessing, validation selection, and scoring reuse the author release. The staging changes export predictions and preserve the KNNCon detector field. The additional mask variant changes textual label conversion from astype(bool) to eq("True") in training and development selection. The wrapper accepts an explicit epoch cap.

code/train.py, code/analyze_results.py, the notebook orchestration, tests, report builder, tables, and analysis are project additions assisted by AI. Training code remains downloaded under .bolt/. Final evidence preserves the successful author-source commands, including their original Colab paths. Those paths are historical records. Active commands resolve paths relative to this project or the selected Colab runtime.

evidence/BOLT_executed.ipynb preserves the actual session history, including debugging and interrupted cells. BOLT_Colab_Final.ipynb is the reorganized entry point with the completed study's intended scope. The final notebook contains no fabricated execution outputs.

## 7 Tests and sanity checks

Run python -m unittest discover -s tests -v, or bash scripts/test.sh. Checks cover unknown-class metric calculations, the string-mask regression, and consistency of all sixteen recorded metric rows with their test predictions. code/author_runner.py check --upstream .bolt checks dataset alignment and patch application after data acquisition.

## Findings and advice for future reproduction

The released KNNCon string conversion admits all nonempty True and False values. At ratio 0.25, partition zero, this gives 2,303 training rows versus 229 explicitly marked rows, and 258 versus 28 familiar development rows. The corrected five-epoch model receives fewer training updates and a smaller validation subset. Interpret the paired score change with those linked effects.

The actual KNNCon training loop runs its full epoch cap despite early-stopping configuration fields. The completed study uses five epochs for the paired comparison. DeepUNK uses its released BiLSTM with GloVe, despite the BERT label in the paper's runtime table. Independent multi-seed and equal-update studies would help separate optimization and supervision effects.

The successful GPU runs used isolated frameworks and PyArrow 17.0.0 with NumPy 1.26.4. GPU checks execute actual operations. If the test fails, inspect the output before launching training. Save results before disconnecting the Colab runtime.

## Folder layout

Report.pdf is the rendered research report. report/ contains editable report text and figures. code/, scripts/, tests/, and pinned requirements define the reproduction. results/ holds saved trial evidence, derived tables, and checksums. Papers/ contains the original study and four significant cited papers. evidence/ contains the executed notebook, three additional cited papers, and the instructor's guidelines. submission-details.json holds fields for the submitting student's identification and agreed contribution.

The contribution table in the report remains unsigned pending author identification and agreement. Rebuild it after entering the actual members and percentages. Runtime logging does not measure human effort or financial charges.

## Rebuilding the report

Install report/requirements.txt in a reporting environment, then run python report/build_report.py. The builder reads results/tables/ and report/Report.md, draws the two figures, and renders Report.pdf. Edit submission-details.json with actual member fields before rebuilding the contribution page. The final PDF was rendered and all eleven pages visually reviewed.

## Public repository and submission files

Repository: https://github.com/emailextra/AI504-BOLT-Reproduction

The final marked revision is tagged submission. To retrieve that revision, use `git clone --branch submission https://github.com/emailextra/AI504-BOLT-Reproduction.git`. Run `python scripts/verify_package.py`, then `python -m unittest discover -s tests -v`. The GitHub workflow repeats the checks on Linux and independently fetches the pinned BOLT data to compare the supervision audit.

Slides.pptx is the editable thirteen-slide presentation. Slides.pdf is the submission export. Each content slide carries its rubric criterion and evidence location. Speaker notes give the fifteen-minute schedule, and presentation/Rehearsal.md contains the live demonstration commands and Q&A preparation.

Run `python scripts/package_submission.py` to generate the required shared-folder structure under submission/ and the Blackboard filenames under submission/Blackboard/. The group-number placeholder is NN until the actual allocation is entered in submission-details.json. The package includes five cited paper PDFs. Member details, agreed percentages, and typed signatures remain for the actual submitting members to complete.
