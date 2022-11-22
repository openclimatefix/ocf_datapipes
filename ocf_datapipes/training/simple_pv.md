# Simple PV

simple_pv.py has a training pipeline for just loading PV data.

The location is chosen using the PV data, PV location data is made.
Then a time is chosen, and PV examples are made.
These examples are then made into batches.

```mermaid
  graph TD;
      A([Load PV data])-->B;
      A-->B1;
      B1[Filter]-->C;
      B[Choose location]-->C;
      C[Location PV data]-->E;
      C-->D[Choose time]-->E;
      E[PV example]-->F;
      F[PV batch];
```
