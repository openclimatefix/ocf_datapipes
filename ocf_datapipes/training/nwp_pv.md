# NWP PV

nwp_pv.py is a training pipeline for loading NWP and PV data.

The location is chosen using the PV data, PV and NWP location data is made.
Then a time is chosen, and PV and NWP examples are made.
These examples are then made into batches and put together into one Batch.

```mermaid
  graph TD;
      A([Load PV data])-->B;
      B[Choose location]-->C;
      A1([Load NWP data])-->B2;
      A-->B1;
      B-->C1
      B1[Filter]-->C;
      B2[Filter]-->C1;
      C[Location PV data]-->E;
      C1[Location NWP data];
      C-->D[Choose time]-->E;
      C1-->D;
      D-->E1;
      C1-->E1;
      E[PV example]-->F;
      E1[NWP example]-->F1;
      F[PV batch] --> G;
      F1[NWP batch] --> G;
      G[Batch];
```
