Compared with the current-state relative-CO baseline and the learned HGNN-risk score,
the proposed low-temperature consensus rescue score improves event-level detection
while maintaining zero warning windows on normal-operation days. Under the selected
operating point, it detects 94.1% of score-evaluable event-rich test events, with a
median lead time of 196.5 minutes and zero normal-operation warning windows.

We additionally perform an event-label-based actionable-opportunity audit. The test
set contains 19 labeled event-rich events. The proposed score detects 18 of these
events overall. The only remaining undetected event is a lead-censored boundary case
with only one minute of valid pre-event context under the actionable-lead definition.
After excluding this boundary-censored case, the proposed score detects 18/18
actionable-evaluable events, while still producing zero warning windows on
normal-operation days.
