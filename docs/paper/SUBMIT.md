# arXiv submission

Upload `arxiv.tar.gz` (built by `python docs/paper/make.py`). It contains `main.tex`, `main.bbl`, `refs.bib`, figures and tables, and builds with pdflatex.

| Field | Value |
|---|---|
| Title | Predicting Where People Look at Ads: An Open Benchmark and a Compact Attention Model Fine-Tuned on Design Eye Tracking |
| Authors | Pratyush Mahapatra |
| Primary category | cs.CV (Computer Vision and Pattern Recognition) |
| Cross-list | cs.HC (Human-Computer Interaction) |
| Comments | 11 pages, 5 figures, 6 tables. Code, model and data pointers: https://github.com/pranikz/MediaAttention |
| Licence | CC BY 4.0 |

## Abstract (plain text, 1551 characters; arXiv allows 1,920)

Predicted-attention tools are widely used to pre-test advertising, yet their accuracy claims are hard to compare and the best open models were trained on natural photographs rather than designed content. We make three contributions. First, we review 15 commercial predictive-attention tools from public sources and show that headline claims of "92-95% accuracy" are mostly AUC expressed as a share of the inter-observer ceiling, a saturated metric on which a centre-only prior already scores 0.78 on MIT300. Second, we benchmark six openly available models, from a 3.7M-parameter network to the 452M-parameter DeepGaze MSDB, against human eye tracking on webpages (FiWI) and on 1,980 posters, app screens and webpages (UEyes), none of which they were trained on. On design content, model size does not buy accuracy: DeepGaze MSDB reaches a correlation (CC) of 0.485 with human fixation maps on UEyes, tied with DeepGaze IIE and below the 3.7M-parameter UNISAL (0.502). Third, we fine-tune UNISAL on the UEyes training split. The resulting model, MediaAttention-Ads v1, reaches CC 0.735 on the held-out UEyes test split and 0.752 on held-out posters, against 0.514 and 0.496 for the best other open model, with non-overlapping 95% confidence intervals, while also improving on out-of-domain webpages (0.631 vs. 0.616). The model runs in 0.04 s per image on a laptop. We release the model under Apache-2.0 together with MediaAttention, an open toolkit that turns attention maps into advertising metrics, and an open browser-based eye-tracking study kit.
