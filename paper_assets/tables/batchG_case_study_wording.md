## Case-study wording draft

### Rescued event 45

Event 45 is the main newly rescued case. In the previous CO-gated HGNN policy, this event was missed, whereas the proposed low-temperature consensus rescue score produces a valid pre-event warning with a lead time of 47.0 min. This case illustrates the value of the mechanism-guided rescue rule: before the event, relative CO had not yet risen substantially, but the learned HGNN/RF signals were already elevated under a low-temperature operating regime. The rescue score therefore captures a precursor pattern that is not well represented by a pure relative-CO gate.

For this event, the pre-event maximum final score is 0.567, while the pre-event maximum relative CO is -0.048. During the event, relative CO reaches 0.793. This suggests that the proposed score can issue an earlier alarm before the large CO excursion becomes obvious.

### Remaining event 52

Event 52 is the only remaining undetected labeled event after the Batch-G policy. The actionable-opportunity audit shows that this event has insufficient valid pre-event context: it is effectively a boundary-censored event. Its maximum valid pre-event score is 0.000, and no valid pre-event alarm is available under the required lead-time constraint. Therefore, we report both the strict score-evaluable result and an opportunity-adjusted audit. The strict result remains conservative, while the opportunity-adjusted audit clarifies that the remaining miss is due to lack of actionable pre-event observation rather than a failure of the warning score on an evaluable event.