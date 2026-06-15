# Step2c Rank-Reversal Presentation Notes

## Main Claim

SPO+ helps in the selected success cases by reranking decision-critical feasible solutions. The 2stage model often has a near-oracle candidate in its top20 list, but ranks it below a worse rank1 decision; SPO+ reverses that ordering. The negative controls show the boundary: reranking fails when no near-oracle candidate is present, or when SPO+ promotes the wrong lower-ranked candidate.

## Mechanism Table

| Graph | Mechanism | Pattern | True delta | 2stage pred delta | SPO+ pred delta | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| G-392 | Deep-candidate correction | helpful_reversal | 61.58 | -4.24 | 3.49 | 2stage has the near-oracle candidate only deep in top20; SPO+ makes it rank1. |
| G-1285 | Clean exact rank-2 promotion | helpful_reversal | 42.02 | -0.60 | 1.72 | 2stage rank2 is the oracle solution; SPO+ promotes it exactly to rank1. |
| G-1560 | Large-effect top-K promotion | helpful_reversal | 70.42 | -2.71 | 3.42 | Largest-effect promotion case; SPO+ promotes a near-oracle top-K candidate. |
| G-1169 | Broad top20 promotion | helpful_reversal | 29.17 | -2.81 | 10.19 | SPO+ selects a near-oracle candidate from deeper in the 2stage top20 list. |
| G-1449 | Deep top20 promotion outside top5 | helpful_reversal | 20.84 | -2.62 | 6.19 | A top5-unexplained case becomes deep top20 promotion after larger-K audit. |
| G-142 | Both-poor negative control | same_rank1_or_no_reversal | 0.00 | 0.00 | 0.00 | No near-oracle candidate is present in 2stage top20; both methods keep the same bad rank1. |
| G-946 | Both-poor negative control | same_rank1_or_no_reversal | 0.00 | 0.00 | 0.00 | No near-oracle candidate is present in 2stage top20; both methods keep the same bad rank1. |
| G-14 | Harmful SPO+ reranking | harmful_reversal | -28.14 | -2.24 | 5.27 | SPO+ promotes a lower-ranked candidate, but that candidate is worse under true labels. |
| G-163 | Harmful SPO+ reranking | harmful_reversal | -44.57 | -2.36 | 0.51 | SPO+ reranks away from a better 2stage decision toward a worse candidate. |

## Main Figure Panels

- Panel A: G-392 (Deep-candidate correction), SPO+ helps; C is 2stage rank 8 and true rank 3.
- Panel B: G-1285 (Clean exact rank-2 promotion), SPO+ helps; C is 2stage rank 2 and true rank 1.
- Panel C: G-1560 (Large-effect top-K promotion), SPO+ helps; C is 2stage rank 2 and true rank 2.
- Panel D: G-14 (Harmful SPO+ reranking), SPO+ hurts; C is 2stage rank 9 and true rank not in top50.

## Top Critical Edges

### G-392
- edge 287 (51->11), added_by_spoplus_rank1, freq=1.00, true_delta=38.65, 2stage_pred_delta=16.64, SPO+_pred_delta=20.57.
- edge 168 (26->44), added_by_spoplus_rank1, freq=1.00, true_delta=34.10, 2stage_pred_delta=15.03, SPO+_pred_delta=19.69.
- edge 278 (50->10), added_by_spoplus_rank1, freq=1.00, true_delta=18.32, 2stage_pred_delta=14.59, SPO+_pred_delta=17.03.
- edge 66 (10->43), added_by_spoplus_rank1, freq=1.00, true_delta=17.06, 2stage_pred_delta=16.57, SPO+_pred_delta=20.04.
- edge 255 (45->24), removed_from_2stage_rank1, freq=1.00, true_delta=-12.60, 2stage_pred_delta=-8.75, SPO+_pred_delta=-14.24.

### G-1285
- edge 70 (19->10), added_by_spoplus_rank1, freq=1.00, true_delta=35.01, 2stage_pred_delta=16.98, SPO+_pred_delta=20.78.
- edge 197 (46->43), added_by_spoplus_rank1, freq=1.00, true_delta=32.66, 2stage_pred_delta=15.00, SPO+_pred_delta=18.87.
- edge 5 (3->7), removed_from_2stage_rank1, freq=1.00, true_delta=-18.21, 2stage_pred_delta=-13.44, SPO+_pred_delta=-15.82.
- edge 35 (13->19), added_by_spoplus_rank1, freq=1.00, true_delta=8.87, 2stage_pred_delta=9.54, SPO+_pred_delta=11.59.
- edge 183 (42->43), removed_from_2stage_rank1, freq=1.00, true_delta=-8.16, 2stage_pred_delta=-10.58, SPO+_pred_delta=-11.67.

### G-1560
- edge 15 (3->5), added_by_spoplus_rank1, freq=1.00, true_delta=41.37, 2stage_pred_delta=17.26, SPO+_pred_delta=21.65.
- edge 228 (42->47), added_by_spoplus_rank1, freq=1.00, true_delta=30.89, 2stage_pred_delta=17.37, SPO+_pred_delta=21.35.
- edge 30 (5->30), added_by_spoplus_rank1, freq=1.00, true_delta=25.10, 2stage_pred_delta=13.63, SPO+_pred_delta=17.09.
- edge 176 (31->42), added_by_spoplus_rank1, freq=1.00, true_delta=24.45, 2stage_pred_delta=14.30, SPO+_pred_delta=16.63.
- edge 257 (50->30), removed_from_2stage_rank1, freq=1.00, true_delta=-21.87, 2stage_pred_delta=-13.83, SPO+_pred_delta=-17.42.

### G-1169
- edge 61 (9->18), added_by_spoplus_rank1, freq=1.00, true_delta=52.30, 2stage_pred_delta=16.78, SPO+_pred_delta=21.01.
- edge 34 (6->4), removed_from_2stage_rank1, freq=1.00, true_delta=-18.64, 2stage_pred_delta=-12.62, SPO+_pred_delta=-17.03.
- edge 159 (23->0), removed_from_2stage_rank1, freq=1.00, true_delta=-17.48, 2stage_pred_delta=-13.26, SPO+_pred_delta=-14.73.
- edge 24 (4->2), removed_from_2stage_rank1, freq=1.00, true_delta=-17.43, 2stage_pred_delta=-12.03, SPO+_pred_delta=-15.43.
- edge 8 (0->45), removed_from_2stage_rank1, freq=1.00, true_delta=-16.00, 2stage_pred_delta=-13.79, SPO+_pred_delta=-15.72.

### G-1449
- edge 176 (44->27), added_by_spoplus_rank1, freq=1.00, true_delta=37.84, 2stage_pred_delta=17.77, SPO+_pred_delta=22.00.
- edge 132 (30->13), removed_from_2stage_rank1, freq=1.00, true_delta=-23.85, 2stage_pred_delta=-14.44, SPO+_pred_delta=-17.53.
- edge 58 (11->43), added_by_spoplus_rank1, freq=1.00, true_delta=11.67, 2stage_pred_delta=9.77, SPO+_pred_delta=13.44.
- edge 177 (44->30), removed_from_2stage_rank1, freq=1.00, true_delta=-5.57, 2stage_pred_delta=-12.95, SPO+_pred_delta=-14.15.
- edge 155 (39->11), added_by_spoplus_rank1, freq=1.00, true_delta=5.21, 2stage_pred_delta=11.95, SPO+_pred_delta=13.54.

### G-14
- edge 73 (11->7), removed_from_2stage_rank1, freq=0.66, true_delta=-39.18, 2stage_pred_delta=-15.97, SPO+_pred_delta=-19.78.
- edge 147 (27->11), removed_from_2stage_rank1, freq=0.66, true_delta=-20.29, 2stage_pred_delta=-13.69, SPO+_pred_delta=-15.62.
- edge 71 (10->36), removed_from_2stage_rank1, freq=0.66, true_delta=-18.69, 2stage_pred_delta=-14.64, SPO+_pred_delta=-17.46.
- edge 177 (36->11), added_by_spoplus_rank1, freq=0.66, true_delta=13.70, 2stage_pred_delta=13.69, SPO+_pred_delta=15.62.
- edge 145 (26->36), added_by_spoplus_rank1, freq=0.66, true_delta=12.67, 2stage_pred_delta=12.03, SPO+_pred_delta=13.18.

### G-163
- edge 346 (52->42), removed_from_2stage_rank1, freq=1.00, true_delta=-42.46, 2stage_pred_delta=-15.87, SPO+_pred_delta=-19.45.
- edge 216 (32->33), removed_from_2stage_rank1, freq=1.00, true_delta=-34.34, 2stage_pred_delta=-15.23, SPO+_pred_delta=-18.19.
- edge 15 (1->29), removed_from_2stage_rank1, freq=1.00, true_delta=-22.44, 2stage_pred_delta=-15.48, SPO+_pred_delta=-18.32.
- edge 70 (9->48), added_by_spoplus_rank1, freq=1.00, true_delta=20.14, 2stage_pred_delta=11.96, SPO+_pred_delta=17.04.
- edge 27 (1->42), added_by_spoplus_rank1, freq=1.00, true_delta=18.58, 2stage_pred_delta=14.66, SPO+_pred_delta=17.49.

