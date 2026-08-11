# E5: Human Evaluation Framework for CoT Explanations

## Objective
Quantify the quality of ViCSynAD's Chain-of-Thought anomaly explanations through
structured human evaluation by domain experts.

## Evaluation Protocol

### Raters
- Minimum 3 raters with cyber-physical systems or industrial process knowledge
- Raters should be blinded to which system generated each explanation
- Inter-rater reliability measured via Fleiss' Kappa

### Materials
Each rater receives:
1. 20 anomaly windows with TS visualization images
2. Corresponding CoT explanations (5-step structured output)
3. Ground truth fault information (when available from SWaT attack scenarios)
4. Scoring sheet (Likert 1-5 scale)

### Rating Dimensions (5-point Likert: 1=Very Poor, 5=Excellent)

| # | Dimension | Question | Anchors |
|---|-----------|----------|---------|
| Q1 | **Pattern Recognition** | How accurately does the explanation identify the visual patterns indicating the anomaly? | 1=Completely misses the pattern, 5=Precisely identifies specific sensors and deviation patterns |
| Q2 | **Causal Plausibility** | How plausible is the identified root cause given the system's process flow (P1→P6)? | 1=Physically impossible causation, 5=Highly plausible causal mechanism consistent with process design |
| Q3 | **Propagation Coherence** | Does the propagation path described make physical sense in the water treatment process? | 1=Violates process flow direction, 5=Perfectly traces physically valid sensor-to-sensor cascade |
| Q4 | **Actionability** | How useful are the recommendations for an operator to take corrective action? | 1=Vague/generic advice, 5=Specific, actionable steps with clear targets |
| Q5 | **Overall Quality** | Overall, how would you rate the quality of this explanation? | 1=Useless, 5=Production-ready |

### Additional Binary Questions
- B1: Does the explanation correctly identify at least one true anomalous sensor? (Yes/No)
- B2: Does the explanation mention any sensor that is clearly NOT anomalous? (Yes/No — reverse scored)
- B3: Would you trust this explanation to guide operational decisions? (Yes/No)

## Scoring Sheet Template

```csv
Window ID, Q1_Pattern, Q2_Causal, Q3_Propagation, Q4_Action, Q5_Overall, B1_Correct, B2_Hallucination, B3_Trust, Comments
001,5,4,3,4,4,Yes,No,Yes,"Good identification of P3 UF feed anomaly"
002,3,2,2,3,2,Yes,Yes,No,"Root cause implausible — claims P6→P3 backflow"
...
```

## Analysis Plan

1. **Descriptive statistics**: Mean, SD, median per dimension
2. **Inter-rater reliability**: Fleiss' Kappa (ordinal weighting) — target κ > 0.6
3. **Dimension correlation**: Pearson r between Q1-Q5
4. **Comparison with baselines**: If we also generate GPT-4 explanations for same windows, compare via paired Wilcoxon test
5. **Hallucination rate**: % of explanations with B2=Yes

## Expected Results (Hypothesis)

Based on the CoT explainer's structured prompt design and causal graph integration:
- Q1 (Pattern): 3.5-4.0 (VLM has strong visual recognition)
- Q2 (Causal): 2.5-3.5 (Limited by causal discovery accuracy)
- Q3 (Propagation): 3.0-4.0 (P&ID prior constrains propagation paths)
- Q4 (Action): 3.0-3.5 (Generic but sensible recommendations)
- Q5 (Overall): 3.0-3.5 (Usable but not production-ready)

## Implementation Notes

1. Use E3 outputs as source material
2. Randomize window order and blind system identity
3. Provide rater training with 2-3 example windows before formal scoring
4. Collect demographic info (years of experience, domain expertise)
5. Platform: Google Forms or Qualtrics for easy distribution
