## Results wording draft

Under the strict score-evaluable event-level policy evaluation, the proposed low-temperature consensus rescue score detects 16/17 event-rich test events, corresponding to an event-level detection rate of 0.941. It achieves a median actionable lead time of 196.5 min, an event-rich warning precision of 0.708, and 1.75 false warning windows per event-rich day. Importantly, it produces zero warning windows on the 19 normal-operation test days.

Compared with the previous CO-gated HGNN score, the proposed score improves event-level detection from 0.882 (15/17) to 0.941 (16/17), while also improving warning precision from 0.667 to 0.708 and reducing event-rich false warning burden from 2.00 to 1.75 false windows per day. Both methods maintain zero warning windows on normal-operation days.

Because some labeled events occur near data/session boundaries, we further conduct an actionable-opportunity audit over all 19 labeled event-rich test events. The proposed score detects 18/19 events overall. The only remaining undetected event is a lead-censored boundary case with insufficient valid pre-event context. After excluding this boundary-censored case, the proposed score detects 18/18 actionable-evaluable events, yielding an opportunity-adjusted detection rate of 1.000.

Suggested one-sentence claim:
The proposed mechanism-guided rescue score improves strict event-level detection from 88.2% to 94.1% while maintaining zero normal-operation warning windows, and achieves 18/18 detection after accounting for lead-censored boundary events.