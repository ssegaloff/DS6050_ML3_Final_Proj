4/10/26 - Sabine

I ran a baseline using pretrained YOLO26l with no fine tuning using baseline_validate.py

I am now going to run an experiment.

So, the experiment is:

1. Baseline — pretrained YOLO26l, no fine-tuning (baseline_validate.py)
2. FREEZE=23 — head only trainable
3. FREEZE=10 — full neck, head, and C2PSA trainable

Runs 2 and 3 use identical hyperparameters, seed, batch size, and epochs (300, patience=20).
Only the freeze point and LR0 differ (0.01329 for run 2, 0.001329 for run 3).

This will help meet the rubric's criteria for Analysis and Interpretation: "discusses the limitations of the analysis" and "analyses map back to the hypotheses." 

I will isolate one variable (C2PSA unfreezing) instead of 2 (C2PSA and SPPF unfreezing) so that I don't confuse what is driving the change.

Each step isolates one decision, which lets us make a direct causal argument in our report about what drove improvement in shark AP50 specifically.

Across runs 2 and 3 we have the same hyperparameters, same seed, same epochs. The only thing changing is the freeze point, and the learning rate. (I am even keeping batch size the same for safety). (Note: LR0 changes to be 10x smaller for freeze=10)