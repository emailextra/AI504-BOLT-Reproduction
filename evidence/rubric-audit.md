# Rubric evidence audit

Checked against the instructor's final submission and presentation guidelines on 18 November 2026. These entries identify evidence and limits; they do not predict a mark.

| Requirement | Evidence and status |
| --- | --- |
| P1 eligible approved paper | BOLT is an ACL Findings 2026 paper. Instructor approval is an administrative item to confirm with the student. |
| P2 specific claim | Report Section 1 gives the four target scores, dataset, class ratios, and label ratio. |
| P3 attempted reproduction | Ten successful DeepUNK trial records, predictions, and metrics. |
| P4 experimental detail | Report Section 3 and README document data, splits, preprocessing, metrics, hardware, and seed. |
| P5 deviations | Report Sections 3.2–3.4 disclose the reduced KNNCon scope, software environment, hardware, and backbone ambiguity. |
| P6 five significant papers | Papers/ contains the original and four cited studies. |
| P7 eight papers and gap | Report Section 2 cites eight studies and identifies the data-selection provenance gap. |
| P8 original contribution | Source-mask audit and evaluated paired correction; quantitative and concrete error analysis. |
| P9 additional contribution | Error taxonomy is evaluated alongside the mask study. Whether this meets the highest level is the marker's judgment. |
| P10 thematic synthesis | Section 2 groups work by novelty definition, representation/rejection, and benchmark comparability. |
| C1 runnable code | Model code ran in Colab. Final wrappers compile; five regression checks and independent data audit pass on a fresh GitHub Linux runner. A fresh GPU rerun of the reorganized entry point remains unperformed. |
| C2 README | All seven required topics appear in order. |
| C3 rerun tolerance | Expected scores and tolerance are documented. Independent model rerun remains unperformed. |
| C4 provenance | README and author_runner.py distinguish reused training implementation and project tooling. |
| C5 command per result | Each report experiment and all derived tables have named commands and shell wrappers. |
| C6 seeds and pins | Seed 2025 fixed, separate framework requirements pinned, recorded package inventory preserved. Both requirements files pin the complete recorded transitive dependency inventories. |
| C7 checks | One unittest command checks metrics, label-mask regression, and prediction integrity. |
| C8 every number reproducible | Saved metrics and measured runtimes trace to scripts and raw evidence. Full model rerun has not been independently repeated. |
| C9 comments | Project functions have purpose docstrings; inherited code is attributed. |
| C10 findings | README records the mask issue, actual stopping behavior, framework setup, limitations, and future checks. |
| R1 section structure | Required sections included. Contribution table is last. AI declaration follows the conclusion. |
| R2 word count | Report text, tables, captions, and bibliography fall in the 4,000–5,000-word band. |
| R3 results | Sections 4–5 provide and interpret results. |
| R4 published comparison | Table 1 uses the selected benchmark ratios and five class partitions. |
| R5 concrete failures | Section 5.2 gives three actual test requests, labels, predictions, and qualified interpretations. |
| R6 limits, ethics, resources | Sections 3.4, 4.3, and 5.3–5.4. |
| R7 citations | Eight cited works each appear in the bibliography and references.bib. |
| R8 discrepancies | Section 5.1 distinguishes established checks from unresolved causal explanations. The 25% unknown-F1 discrepancy remains unexplained causally. |
| R9 sensitivity | Known-class-ratio variation and paired supervision-mask analysis are reported. Fold SD is correctly distinguished from seed variation. |
| R10 verdict | Partial support is stated in abstract, discussion, and conclusion. |
| R11 figures | Two figures and three tables are referenced and interpreted. |
| R12 new observation | The released Boolean conversion and unused early-stop setting are documented from the inspected source. |

Three of four DeepUNK comparisons fall within two points of the paper. The fourth differs by 2.97 points. The study therefore does not establish all the report rubric's Excellent performance conditions, and it makes no claim of exceeding published results. Successful reproduction of an observed subset remains scientifically valid.

The user confirmed Group 4, all four member names and IDs, equal 25% shares, the listed roles, and typed agreement names. Instructor paper approval remains an administrative matter outside the experiment evidence.


## Final submission preparation

The public repository is https://github.com/emailextra/AI504-BOLT-Reproduction. Its submission tag identifies the revision for marking. Report.pdf links to the repository on the first page. Slides.pptx and Slides.pdf contain thirteen slides with evidence-pointer footers and a fifteen-minute rehearsal schedule in the speaker notes. presentation/Rehearsal.md covers the live demonstration and individual discussion. scripts/package_submission.py creates the requested folder and Blackboard filenames.

Group 04 filenames, confirmed member identities, equal contributions, and typed agreements are complete. The presentation notes allocate at least three minutes to every member within fifteen minutes. Actual presentation delivery, Q&A, and grading occur in the course session.
