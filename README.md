# DAMG6105 original coursework snapshot

Coursework by **Zainab Cheema and Jiabei Liu**. This repository preserves the original diabetes-clustering assignment for historical reference. The actively documented and tested version is [Glucose Pseudo-Label Evaluation](https://github.com/jiabeiliu/glucose-pseudolabel-evaluation).

The dataset files here are byte-for-byte identical to the CSV files in the improved repository, despite their different filenames. The improved version splits data before fitting preprocessing, makes the generated-label interpretation explicit, and includes tests. Do not present these as separate research projects.

Important limitation: the labels are generated from K-means clusters, not observed diabetes outcomes. Any reported accuracy measures agreement with those generated labels; it is **not** diagnostic accuracy. This original snapshot is not recommended as a runnable portfolio demo because its README and preprocessing order were superseded by the improved version.
