# Reproducing BOLT under Limited Supervision for Banking Intents

AI504 Introduction to Computational Linguistics  
Assignment 2 Reproduction and validation study  
Experiment dates 17–18 November 2026  


## Abstract

Open-set intent classification requires a model to recognize supported requests while rejecting requests from unfamiliar intent categories. This study reproduces a focused part of BOLT using BANKING77 and investigates whether released code respects its nominal supervision budget. DeepUNK was evaluated on five official class partitions at known-class ratios of 0.25 and 0.50, with a labeled-data ratio of 0.10 and seed 2025. Mean known-class F1 was 36.42 and 47.77, compared with published values of 37.49 and 48.99. Mean unknown-class F1 was 1.19 and 32.99, compared with 4.16 and 31.21. Three of these four comparisons differ by fewer than two percentage points. An independent prediction audit reproduced every saved metric within rounding tolerance. The code inspection additionally identified a string-to-Boolean conversion that admitted rows marked False into KNNCon training and validation. A paired five-epoch, single-partition study compared the released conversion with an explicit True comparison. With cosine-distance local outlier detection, known-class F1 fell from 69.80 to 11.60 and unknown-class F1 from 81.59 to 14.34 after correction. This difference combines reduced data access, fewer optimizer updates, and changed validation selection. The evidence partly supports the selected DeepUNK results and establishes a consequential implementation discrepancy, while the restricted KNNCon study supports a qualified conclusion about the released supervision protocol.

## 1 Introduction

Banking assistants must interpret short requests about payments, account access, identity verification, and currency exchange. A deployed assistant can encounter a sensible banking question whose intent is absent from its supported inventory. Assigning such a request to a familiar intent can send the customer through an irrelevant process. Open-set classification therefore combines recognition of known categories with rejection of unfamiliar examples. This creates an evaluation problem in which classification accuracy alone can conceal failures to recognize the boundary of the system's competence.

BOLT supplies a recent framework for studying this problem across datasets and supervision conditions (Qin et al., 2026). The present study selects its BANKING77 open-set experiments and examines one released implementation detail. The central numerical target is the reported DeepUNK change from 37.49 to 48.99 known-class F1 and from 4.16 to 31.21 unknown-class F1 when the known-class ratio increases from 0.25 to 0.50 at a labeled-data ratio of 0.10. The experiment tests these four values using the released data and five partitions, with predictions retained for independent verification.

The research questions are whether this focused result can be recovered using the public implementation, how strongly the result depends on the chosen class partition, and whether a nominal low-supervision configuration actually restricts access to labeled examples. These questions relate directly to experimental validity. A setting named as ten-percent supervision should select the corresponding rows before the model learns from their labels. If selection differs from that definition, a score can be numerically correct for the executed program while describing a different experiment from the intended protocol.

The study makes two evaluated contributions beyond executing the release. It audits the low-supervision mask against the stored row selections and measures a paired KNNCon correction. It also separates errors into unknown acceptance, known rejection, and confusion between familiar categories, using saved test predictions. These analyses make the consequences of the implementation choices visible and connect aggregate metrics to concrete banking requests.

The scope was chosen to fit available resources. DeepUNK receives a complete five-partition evaluation at both selected ratios. KNNCon receives a five-epoch comparison on one partition with the released and corrected masks. This narrower comparison cannot establish its published five-partition performance, and the conclusions retain that limitation. Section 2 relates the design to prior research. Sections 3 and 4 document methods and outcomes. Section 5 interprets discrepancies, uncertainty, errors, and resources, followed by a verdict and reproducibility information.

## 2 Related Work

### 2.1 Defining unfamiliar inputs

Open-set text learning requires a definition of what counts as unfamiliar. Shu et al. (2017) address documents whose classes were absent during training through DOC, an early deep open-classification method. Their formulation makes rejection an explicit part of text classification. This motivates examining the unknown decision independently of the model's ability to distinguish familiar labels. A classifier can maintain useful separation between known categories and still make unreliable rejection decisions near their boundaries.

Larson et al. (2019) introduce an intent dataset with explicit out-of-scope requests. Its evaluation highlights that supported-intent classification and unsupported-query detection place different demands on a system. Their data spans many domains, whereas the present experiment holds out categories within banking. The two designs expose different forms of novelty. A request about an unrelated activity may differ lexically from training material, while an unfamiliar banking intent can share words with familiar payment or account questions. This distinction matters when interpreting false acceptance in the current results.

Casanueva et al. (2020) provide BANKING77 alongside efficient intent detectors based on sentence encoders. The fine-grained banking inventory is useful for testing semantic discrimination within one application domain. In this study, category membership determines whether an example is known, so unfamiliarity is relative to the selected label list. A sentence does not become intrinsically meaningless because its intent is held out. The benchmark changes the model's permitted knowledge of the inventory while retaining plausible customer language.

### 2.2 Representation learning and rejection

DeepUNK combines a BiLSTM feature extractor trained with a margin objective and local outlier detection (Lin and Xu, 2019). The feature objective encourages discrimination, while the density method identifies examples that appear unusual relative to familiar training representations. This separation provides a useful diagnostic structure. Errors may arise because the representation places a novel request near familiar examples, because a familiar request appears atypical, or because the rejection rule handles that geometry poorly.

KNNCon instead uses neighborhood-based contrastive learning to shape representations for out-of-domain intent classification (Zhou et al., 2022). Its relevance here is that neighbor relationships depend on which examples enter training. Accidentally increasing supervised access changes the information available for representation learning. It also changes the number of batches in an epoch. A correction to data selection therefore affects both the learning problem and the optimization budget. The paired comparison in this study evaluates the practical consequence of that combined change.

The two methods share a broad dependence on feature geometry, but they obtain that geometry through different initialization and training choices. The released DeepUNK uses GloVe embeddings, whose global co-occurrence training supplies lexical vectors (Pennington et al., 2014). BERT instead supplies contextual representations that can be fine-tuned for downstream tasks (Devlin et al., 2019). These resources help explain why identical label budgets need not yield identical learning behavior. The current experiment evaluates released systems as configured and avoids attributing their differences to architecture alone.

### 2.3 Benchmark comparability and the remaining gap

BOLT brings datasets, class ratios, label ratios, and multiple open-world methods into a common evaluation framework (Qin et al., 2026). This standardization is valuable because a result depends on the inventory of known classes, how supervision is sampled, and the metric used to pool unfamiliar categories. Consistency at the configuration level still requires consistency in execution. A data mask is part of the protocol and deserves the same scrutiny as a learning rate or model checkpoint.

Together these eight studies connect three layers of the problem. DOC and the out-of-scope evaluation establish the need for rejection. BANKING77 makes that problem fine-grained within a customer-service domain. DeepUNK and KNNCon provide distinct mechanisms for learning and rejecting, while GloVe and BERT identify their representation resources. BOLT supplies the controlled benchmark in which these choices can be compared. This synthesis motivates auditing the entire route from configuration to examples, representation, prediction, and final score.

The specific gap examined here concerns supervision provenance in the released pipeline. The literature's algorithmic descriptions do not establish which rows a particular checkout actually selects when it reads a Boolean mask as text. The present contribution tests that implementation behavior directly and measures its effect under a disclosed budget. The resulting evidence is relevant to reproducing this checkout and to interpreting low-label comparisons. It does not imply that all implementations of the underlying KNNCon method share the same defect.

## 3 Methodology

### 3.1 Data and experimental units

The study uses the BANKING77 files distributed with BOLT at commit 2c935438b5506febcfd23e499e3de81868236b17. Original train, development, and test TSV files are combined with the stored 0.10 labeled-data masks and fold5 label lists. Row order and labels are checked before masks are associated with text. The test set contains 3,080 requests. Labels outside the selected known inventory are mapped to one unknown class for evaluation. The underlying BANKING77 source is credited to Casanueva et al. (2020), and its dataset card lists the CC BY 4.0 licence.

The known-class ratios select 19 and 38 familiar intents from the 77-category inventory. Five supplied class partitions are evaluated independently at each ratio. These partitions should be understood as changes in class composition. Their mean summarizes performance over the selected benchmark partitions, and their standard deviation measures partition variation. Seed 2025 is fixed throughout. A five-partition standard deviation is consequently reported with its actual source of variation rather than described as uncertainty over independent random seeds.

At ratio 0.25 in partition zero, the stored training mask identifies 229 labeled familiar requests from 2,303 eligible familiar requests. The development counterpart contains 28 marked rows from 258 familiar requests. At ratio 0.50, the corresponding partition has 472 marked training rows from 4,740 and 55 development rows from 527. These counts are derived from the released data and provide a direct test of the intended supervision restriction.

### 3.2 Systems and training

DeepUNK is executed through the released TensorFlow implementation with 300-dimensional GloVe embeddings and a BiLSTM feature extractor. The runner supplies a 50-epoch cap, following the benchmark grid protocol, and retains the released validation-loss early-stopping behavior. It selects the best saved checkpoint before feature extraction and local outlier evaluation. The implementation pads sequences to 4,096 positions; that released setting was retained in the completed GPU evidence.

KNNCon uses the released PyTorch code and the google-bert/bert-base-uncased checkpoint, a batch size of 16, maximum sequence length 128, learning rate 0.00005, and contrastive training. Both paired runs have a five-epoch cap and use partition zero at ratio 0.25. The training loop retains its best validation model. Inspection confirms that this loop iterates to the epoch cap despite configuration fields naming early stopping. The comparison therefore uses the actual implemented stopping behavior.

The paired variants preserve the same seed and epoch count. The released variant converts nonempty text mask values with astype(bool), causing both True and False strings to pass. The corrected variant uses eq("True") for training and familiar-development selection. This design tests the consequences of enforcing the supplied mask in both places. It does not isolate training-data restriction from validation-data restriction, and the different dataset sizes imply different numbers of parameter updates.

### 3.3 Evaluation and verification

Known-class F1 is the macro average of per-class F1 over familiar labels. Unknown-class F1 treats all unfamiliar intents as a pooled class. Overall macro F1 and accuracy are also retained. All report metrics use a zero-to-one-hundred scale. The analysis converts DeepUNK's fractional outputs to percentages and preserves KNNCon's already scaled outputs. This normalization is necessary before comparing the saved outputs or calculating differences.

For DeepUNK, five fold values are averaged at each ratio, with sample standard deviation across partitions. For KNNCon, cosine and Euclidean local outlier variants are reported separately. The source evaluates both variants, while the benchmark appendix supplies one aggregate KNNCon entry. Choosing whichever variant best matches the test score would introduce an additional selection decision, so both executed outputs remain visible. The primary paired discussion uses the cosine variant with the Euclidean outcome provided as a complementary observation.

A separate audit recalculates all saved metrics directly from predictions. Every score row contains 3,080 test predictions, and recomputed values agree with stored scores to within 0.011 percentage points, covering rounding. A hand-calculated metric test checks the scaling and unknown-class treatment. The package also includes a regression check for the textual mask behavior, scripts for regenerating tables, and a reduced model run. Test labels are used for evaluation and error inspection after prediction.

### 3.4 Deviations and implementation provenance

Published training and scoring code is reused with prediction-export instrumentation, retention of the KNNCon detector field, and a separate corrected-mask variant. The original work contributed by this study comprises protocol checks, data counts, paired execution, independent metric verification, tabulation, and error analysis. These changes are recorded in the source files and README.

Hardware differs from the authors' reported workstation. The completed runs use a Colab Tesla T4 with 15,360 MiB device memory and isolated Python 3.11 environments for TensorFlow and PyTorch. GPU operations were checked in both environments, including an LSTM gradient computation. PyArrow was pinned to version 17.0.0 for NumPy 1.26.4 compatibility. The narrower KNNCon scope, fixed seed, framework isolation, and package pins are disclosed so the implemented experiment can be reproduced.

## 4 Experiments and Results

### 4.1 DeepUNK reproduction

Table 1 compares the five-partition means with the published appendix values. Known-class F1 increases from 36.42 at ratio 0.25 to 47.77 at ratio 0.50. The differences from the published scores are negative 1.07 and negative 1.22 percentage points. Unknown-class F1 increases from 1.19 to 32.99, differing by negative 2.97 and positive 1.78 points. Thus three values satisfy a two-point proximity criterion. The remaining gap is the unknown-class metric under the smaller familiar inventory.

[TABLE_DEEPUNK]

The partition-level variability qualifies this agreement. Unknown-class F1 at ratio 0.50 has a standard deviation of 12.49 points, while known-class F1 at that ratio varies by 2.59 points. The observed mean therefore combines much more variable rejection behavior than familiar-class recognition. Figure 1 displays the fold values alongside the published references. Reporting only the averages would conceal the uneven behavior across class inventories.

[FIGURE_DEEPUNK]

These results support the direction of the selected DeepUNK change as the known-class ratio increases. They do not establish that unknown-class F1 must always decrease when more classes become familiar. In this subset of the benchmark, both published and reproduced DeepUNK unknown-class F1 increase. The conclusion is tied to this method and dataset, rather than applied to the whole benchmark or its large-language-model results.

### 4.2 Supervision-mask sensitivity

Table 2 presents the paired KNNCon scores. With cosine local outlier detection, the released mask achieves 69.80 known-class F1, 81.59 unknown-class F1, and 75.75 accuracy. The explicit-mask variant achieves 11.60, 14.34, and 13.60 respectively. The corresponding changes are negative 58.20 points for known-class F1 and negative 67.25 points for unknown-class F1. The difference establishes that the executed pipeline is highly sensitive to the supervision selection at the tested budget.

[TABLE_KNNCON]

The Euclidean variant also deteriorates, with known-class F1 changing from 58.04 to 12.87 and unknown-class F1 from 57.36 to zero. Figure 2 shows both detector variants. These outcomes are single-run observations. They measure the consequence of applying the mask under the five-epoch protocol, including the smaller number of updates and the changed development subset. They cannot be read as a causal estimate of the effect of extra labels in isolation.

[FIGURE_MASK]

The released cosine result is higher than BOLT's published KNNCon values of 63.07 known-class F1 and 67.07 unknown-class F1 at this nominal setting. That comparison is contextual because the present KNNCon run uses one partition and a shorter cap. It provides no basis for claiming that this study exceeds the published method. The defensible conclusion is that this released filter and this corrected filter execute materially different supervision conditions.

### 4.3 Cost and saved evidence

The ten completed DeepUNK trials required 1,198.30 seconds, approximately 19.97 GPU minutes. The released five-epoch KNNCon trial required 620.76 seconds, and the corrected trial 100.80 seconds. The reduced one-epoch KNNCon trial completed in 176.59 seconds, under the required thirty-minute demonstration budget. The final experiment total is approximately 32.00 minutes, excluding the reduced run, downloads, setup, failed attempts, and the interrupted longer KNNCon run.

Every completed trial retains its invocation, elapsed seconds, console log, metrics, and predictions. The aggregate tables are rebuilt by the analysis command from these files. The evidence audit verified sixteen metric rows, including two detector outputs for each KNNCon run. Runtime differences are also consistent with the smaller number of admitted examples, although they include framework startup and evaluation and cannot be converted directly into per-update speed differences.

## 5 Discussion

### 5.1 Verdict and explanations of discrepancies

The selected DeepUNK finding is partly supported. Its known-class scores are close to the published values, and its change with the known-class ratio has the same direction. The unknown-class result at ratio 0.25 remains outside the two-point threshold. At that low absolute score, a difference of a few points represents a large relative change. The study therefore reports the absolute gap and avoids treating relative percentage error as a useful summary.

Several observations constrain explanations. The scripts use the stored benchmark files and fixed label partitions, ruling out an intentional change to the test inventory. The independent metric audit rules out a scale-conversion mistake or mismatch between saved predictions and reported metrics. The retained GPU logs establish that the completed runs executed successfully. However, these checks do not recover the authors' exact framework environment, initialization stream, checkpoint history, or undocumented choices.

The remaining explanations are consequently hypotheses. Numerical differences between software versions, initialization, and validation checkpoint selection could affect features and outlier decisions. The large partition variation in unknown-class F1 shows that the rejection outcome is sensitive to which intents are familiar. Establishing which mechanism explains the specific published gap would require additional controlled runs or the authors' original per-partition predictions. The current evidence supports a documented partial reproduction with an unresolved discrepancy.

The KNNCon study sharpens a different conclusion. Its released text conversion fails to implement the stated labeled-row filter. The data audit proves that discrepancy before considering model scores. The paired runs show that enforcing the supplied mask changes the evaluated system substantially. Since both training and validation restrictions change together, and five epochs contain far fewer updates after correction, the experiment identifies the practical combined effect. A future factorial study could hold validation selection or update count constant to separate those mechanisms.

### 5.2 Error analysis

The error taxonomy distinguishes three failures that have different practical meanings. Unknown acceptance assigns a held-out intent to a familiar class. Known rejection declines a request the system was trained to support. Wrong-known classification chooses an incorrect familiar label. Correct predictions form the fourth accounting category. Counts cover all 3,080 test rows in each trial, ensuring that the taxonomy can be checked against accuracy.

For the released five-epoch cosine KNNCon run, 2,333 examples are correct, 691 unfamiliar requests are accepted, 44 familiar requests are rejected, and 12 familiar requests receive the wrong familiar label. Correcting the mask reduces the correct count to 419. Table 3 reports all paired counts and provides evidence that the aggregate F1 drop corresponds to extensive changes in individual predictions. These counts measure behavior within the benchmark inventory and should not be translated directly into financial loss or customer harm.

[TABLE_ERRORS]

Three concrete DeepUNK failures in partition zero illustrate the taxonomy. The request "Can I exchange currencies with this app?" has the held-out exchange_via_app intent but is predicted as the familiar fiat_currency_support intent. Shared currency vocabulary provides a plausible explanation for acceptance. The distinction concerns performing an exchange versus asking which currencies are supported. The model's familiar prediction is linguistically related while still being wrong under the held-out inventory.

The request "What countries can I travel to and have access to their currencies?" is labeled fiat_currency_support but receives beneficiary_not_allowed. Both labels belong to familiar categories in this partition, so this is a wrong-known error. The example suggests that the short sentence's travel and currency language does not reliably preserve the intended semantic distinction in the learned representation. That interpretation is based on the text and output; token-level attribution was not measured.

The request "Where are my funds being sourced from?" belongs to the familiar verify_source_of_funds category but receives the unknown decision. This is a false rejection of a supported request. Its wording may be atypical relative to the training examples or fall outside the learned density region. Comparing the sentence with its closest training neighbors would be a useful further diagnostic. The three examples establish concrete error types while keeping the explanation of their causes appropriately tentative.

### 5.3 Limitations and ethical considerations

The main scope limitation is coverage. Only BANKING77 is evaluated, and the experiment addresses open-set classification rather than discovery of distinct new categories. English customer requests in one domain provide limited evidence about other languages, longer documents, or real deployment traffic. The study also uses the authors' class inventories, so it assesses those designed scenarios rather than naturally changing customer demand.

There is one random seed. Five DeepUNK partitions measure variation in class composition, while the KNNCon sensitivity study has one partition. Neither permits a seed-based confidence interval for the mask effect. The five-epoch cap may particularly disadvantage the corrected variant because it sees roughly a tenth as many selected examples per epoch. This is a meaningful property of the practical budget but limits a claim about attainable corrected performance after convergence.

Unknown-class F1 is affected by the composition and prevalence of the pooled unknown class. Changing the familiar inventory alters that prevalence and the examples being rejected. An apparent improvement can reflect different class geometry or base rates as well as a better rejection boundary. Future comparisons should retain class-wise counts and examine precision and recall separately. The present report therefore provides the class ratios, test count, individual predictions, and error categories needed to inspect those effects.

Ethically, banking text can concern identity verification and access to funds. A system that confidently routes an unsupported request may obscure its limitations, while excessive rejection can frustrate customers whose intents are supported. Evaluation should distinguish these outcomes and support a fallback to a qualified service channel. The experiment uses the public benchmark data without collecting customer records, demographic profiles, or account information. It provides no subgroup fairness assessment because those attributes and associated ground truth are unavailable.

### 5.4 Reproducibility and resource interpretation

The practical work exposed two reproducibility concerns beyond numerical agreement. A runnable configuration can misstate the effective supervision budget, and a framework callback can be configured without being used by the actual training loop. Both observations justify inspecting selection and stopping behavior instead of inferring execution solely from YAML fields. The release's backbone description also varies between its runtime table and the inspected DeepUNK implementation. The report identifies the BiLSTM system actually executed.

Environment preparation required separate PyTorch and TensorFlow installations, GPU validation, and a compatible PyArrow pin. Earlier CPU and interrupted runs informed debugging but are excluded from the final metric summaries. The surviving evidence records the successful GPU trials and their commands. Exact development labor and financial compute charges were not metered. Runtime accounting therefore covers measured successful processes and explicitly identifies its exclusions.

The package offers scripts to verify predictions, regenerate tables, rerun the ten DeepUNK trials, execute the short paired study, and perform a reduced demonstration. Existing completed trials are skipped only when their completion record exists. A user can select a new results directory for independent model execution. This design preserves the submitted evidence while giving another researcher a route to inspect and challenge every reported result.

## 6 Conclusion

This study reproduced a bounded BANKING77 component of BOLT and audited the supervision protocol in its released KNNCon pipeline. DeepUNK's five-partition known-class F1 increased from 36.42 to 47.77 as the familiar inventory grew, closely tracking the published values. Unknown-class F1 increased from 1.19 to 32.99. The smaller-inventory score remained 2.97 points below the published value, while the other three comparisons differed by fewer than two points. The selected performance claim is therefore partly supported, with rejection behavior requiring particular caution.

The original code audit showed that textual False mask values were admitted as supervised examples. The paired KNNCon experiment demonstrated a substantial performance change when the supplied training and validation masks were enforced. Its restricted setting supports a conclusion about the implemented supervision condition and the tested optimization budget. It leaves the relative roles of training access, validation selection, and update count unresolved.

Independent verification connected all sixteen saved metric rows to their underlying predictions, and the error taxonomy connected model scores to acceptance, rejection, and semantic-confusion failures. Further work should compare equal update budgets, separate training and validation corrections, and evaluate multiple seeds and class inventories. The resulting package supplies the data provenance, scripts, raw evidence, and qualified conclusions needed to extend those checks.

## AI use declaration

OpenAI Codex assisted with literature searches, code inspection, experiment implementation, analysis, and report drafting. Google Gemini assisted with troubleshooting the Colab environment. Reported numerical results were verified against saved experiment outputs.

## References

[1] Chuan Qin, Xi Chen, Jinpeng Li, and Hengshu Zhu. 2026. BOLT: Benchmarking Open-World Learning for Text Classification. Findings of the Association for Computational Linguistics: ACL 2026, 13624–13658. https://doi.org/10.18653/v1/2026.findings-acl.667

[2] Lei Shu, Hu Xu, and Bing Liu. 2017. DOC: Deep Open Classification of Text Documents. Proceedings of EMNLP 2017, 2911–2916. https://doi.org/10.18653/v1/D17-1314

[3] Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. 2019. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. Proceedings of EMNLP-IJCNLP 2019, 1311–1316. https://doi.org/10.18653/v1/D19-1131

[4] Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić. 2020. Efficient Intent Detection with Dual Sentence Encoders. Proceedings of NLP4ConvAI 2020, 38–45. https://doi.org/10.18653/v1/2020.nlp4convai-1.5

[5] Ting-En Lin and Hua Xu. 2019. Deep Unknown Intent Detection with Margin Loss. Proceedings of ACL 2019, 5491–5496. https://doi.org/10.18653/v1/P19-1548

[6] Yunhua Zhou, Peiju Liu, and Xipeng Qiu. 2022. KNN-Contrastive Learning for Out-of-Domain Intent Classification. Proceedings of ACL 2022, 5129–5141. https://doi.org/10.18653/v1/2022.acl-long.352

[7] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. Proceedings of EMNLP 2014, 1532–1543. https://doi.org/10.3115/v1/D14-1162

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019, 4171–4186. https://doi.org/10.18653/v1/N19-1423

## Contribution statement

Group 4 confirms the responsibilities and equal contribution shares shown below. Each member's typed name records agreement with this statement. The README and source files identify reused code and AI assistance.

[TABLE_CONTRIBUTION]
