# Frozen-checkpoint copy experiment

This experimental head uses the final normalized representation at an answer position to assign attention weights to earlier positions. It sums attention on occurrences of the candidate token and mixes that copy probability with the ordinary model probability using a learned sigmoid gate. All Franok decoder weights are frozen.

On the v0.2 checkpoint, fitting the head to 2,220 records and selecting on 180 validation records moved average assistant-token loss from 2.24817 to 2.24750. The selected head placed roughly one percent of probability mass on the copy mechanism. On a separate set of 80 bilingual Belebele questions, both versions scored 23 correct. A fresh 16-check Q4 conversation script moved from 3 to 4 checks passed. The public chat release uses the ordinary decoder.

Reproduce the fit after preparing the [v0.2 corpora](../../docs/data.md).

```sh
python -m experiments.pointer.train --checkpoint work/v2/run --data work/extended/data
```
