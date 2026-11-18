# Presentation rehearsal and individual discussion

The PowerPoint contains thirteen slides and speaker notes timed to fifteen minutes. The PDF is the slide submission copy. Enter the actual group number, names, and student IDs on the opening slide when those details are available.

| Block | Slides | Time |
| --- | --- | --- |
| Opening | 1 | 1 minute |
| Project | 2–5 | 5 minutes |
| Code | 6–8 | 4 minutes |
| Report | 9–12 | 4 minutes 30 seconds |
| Closing | 13 | 30 seconds |

In PowerPoint, use Presenter View to read the notes. Allocate the slides so every member speaks for at least three minutes. Rehearse the complete presentation with a timer. The Code block includes time for showing a live command or an authentic screen recording.

## Live demonstration

Clone the repository and use Python 3.11. Run `python code/analyze_results.py` and open `results/tables/deepunk_comparison.csv`. Show `results/tables/prediction_audit.csv`, then run `python -m unittest discover -s tests -v`. These commands recompute the report evidence from predictions without downloading models.

For the reduced model demonstration, prepare a Colab GPU session and use `python code/train.py quick --output results/demo` from the pinned PyTorch environment. The recorded model run took 176.59 seconds on a T4. Rehearse its timing before using it live. Use a new output directory if you intend to train again, because a completed trial is otherwise skipped. The successful original log remains in `results/quick/knncon_0.25_0/`.

## Questions every member should be ready to answer

- What is known or unknown in this benchmark? The distinction depends on the selected intent inventory, with held-out intents pooled into one unknown class.
- What does DeepUNK train? The inspected release trains a GloVe-initialized BiLSTM and uses local outlier detection on its representations.
- What does KNNCon train? It fine-tunes BERT using neighborhood contrastive training and evaluates local outlier detection.
- What does the label-mask bug do? Text values True and False are both nonempty, so astype(bool) admits both. An explicit equality comparison respects the mask.
- Why do the corrected scores fall? The comparison changes labeled training access, validation access, and optimizer-update count together. The experiment does not separate their effects.
- Why report both LOF distances? The executed evaluator emits both. Selecting a distance based on its test result would add an unjustified model-selection decision.
- What does fold standard deviation mean? It measures variation over the five fixed class partitions, with the seed held at 2025.
- Did the claim reproduce? It is partly supported. Three of four DeepUNK target comparisons are within two percentage points, while the 25% unknown-class result differs by 2.97 points.
- Where is an example failure? Test rows 0, 2, and 41 of DeepUNK partition zero correspond to the three cases in Report §5.2.
- Can you prove every displayed number? Open the referenced report table, raw prediction file, saved run.json, or generating command in the slide footer.

The instructor may select who answers and ask about another member's part. Actual members should review the code and their agreed contribution before the session.
