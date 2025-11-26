# Goal
Compare Adam vs SPSA optimization techniques on Reverse Task on two metrics: minimal convergence time and out-of-distribution (OOD) behavior. This will be tested across a variety of sequence lengths. All examples will use a vocab size of 60 content tokens and 4 administrative tokens for a total of 64 tokens. Multiple data points should be taken and aggregated for each comparison point in the end.

## Evaluating Convergence Time
For the first step, of evaluating convergence time, there are multiple hyperparameters (HPPs) that can be tuned which affect the convergence behavior. Ideally, we would train models across a variety of sequence lengths, ranging from something like 10 to 1000 (10 ** np.arange(1,3.2,0.2)). The tuning of these HPPs also sets the stage for Part 2, Evaluating OOD Behavior.

We want to tune these HPPs across a variety of sequence lengths. This leads to the challenge of finding optimal values (for both SPSA and Adam) across this variety of sequence lengths. It also suggests the problem: should the HPPs be the same across sequence lengths?

## Evaluating OOD Behavior
When it comes to evaluating OOD behavior, and evaluated across sequence lengths in that region and up to double it, or however far it takes to start seeing substantial loss penalties. There is this concept of the minimum unintelligent loss which is the loss associated with the distribution of all tokens. In other words, if we model the model as memorizing the first K tokens and reporting randomly for the rest of the tokens, then we get a particular loss curve for a model trained on N tokens and asked to calculate the result for M tokens. If we plot the real (N, M) loss curve against different values of K, can we reliably estimate the corresponding value of K? (Is one question.)
