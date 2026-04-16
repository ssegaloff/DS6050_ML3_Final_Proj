4/10/26; 4/16/26 - Sabine

I ran a baseline using pretrained YOLO26l with no fine tuning using baseline_validate.py

I am now going to run an experiment.

So, the experiment is:

1. Baseline — pretrained YOLO26l, no fine-tuning (baseline_validate.py)
2. FREEZE=23 — head only trainable
3. FREEZE=11 — upper neck + head trainable (SPPF and C2PSA remain frozen)
4. FREEZE=10 — C2PSA + upper neck + head trainable (SPPF remains frozen)

Runs 2, 3 and 4 use identical hyperparameters, seed, batch size, and epochs (300, patience=20).
Only the freeze point and LR0 differ (0.01329 for run 2, 0.001329 for runs 3 and 4).

This will help meet the rubric's criteria for Analysis and Interpretation: "discusses the limitations of the analysis" and "analyses map back to the hypotheses." 

In run 3, we additionally unfreeze the upper neck (upsample/concat/C3k2 blocks). In run 4, we additionally unfreeze C2PSA. We are doing this to isolate one variable (C2PSA unfreezing) instead of 2 (C2PSA and SPPF unfreezing (or neck vs neck and C2PSA)) so that I don't confuse what is driving the change. Note: SPPF remains frozen in both runs 3 and 4.

Each step isolates one decision, which lets us make a direct causal argument in our report about what drove improvement in shark AP50 specifically.

Across runs 2, 3, and 4 we have the same hyperparameters, same seed, same epochs. The only thing changing is the freeze point, and the learning rate. (I am even keeping batch size the same for safety). (Note: LR0 changes to be 10x smaller for freeze=10 and freeze=11)

I did this once where each training was from COCO weights (sharks_v5...). I will subsequently be doing a series of runs where we use the weights from freeze=23 for our train with freeze=11, and the output weight of that with our train with freeze=10.

I have begun sharks_v6:
- I took the model weights from sharks_v5_freeze23 (trained from COCO, all but head frozen) and am training sharks_v6_freeze11 which has unfrozen the upper neck layers (upsample/concat/C3k2 blocks) in addition to the head. SPPF and C2PSA remain frozen.
- Then I will take the model weights from sharks_v6_freeze11 and train sharks_v6_freeze10 which unfreezes the neck, head, and the C2PSA layer. SPPF remains frozen.