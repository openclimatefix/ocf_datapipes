# Example training datapipes

These are example datapipes using the `ocf_datapipes` library

## Simple PV

`simple_pv.py` has a training pipeline for just loading PV data.

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


## GSP, PV, Satellite, and NWP Pipeline

`gsp_pv_satellite_nwp.py` is a training pipeline for loading GSP, PV, and Satellite and NWP.

The location is chosen using the center of the GSP location and using GSP timestamps for forecast
time t0. Then a time is chosen, and PV, Satellite and NWP examples are made.

```mermaid
graph TD
    A1([Load PV])
    A2([Load GSP]) --> B
    B[Select Location]
    A3([Load NWP])
    A4([Load Satellite])
    A1 --> C1
    A2 --> C2
    C1[PV location]
    C2[GSP location]
    C3[NWP location]
    C4[Satellite location]
    B --> C1 --> D1
    B --> C2 --> D1
    B --> C3 --> D1
    B --> C4 --> D1
    A3 --> C3
    A4 --> C4
    D1[Select Joint Time Periods] --> D2
    D2[Select T0]
    C1 --> E1
    C2 --> E2
    D2-->  E1[PV Time and location]
    D2 --> E2[GSP Time and location]
    D2 --> E3[NWP Time and location]
    D2 --> E4[Satellite Time and location]
    C3 --> E3
    C4 --> E4
    E1 --> F1[PV Batch] --> G
    E2 --> F2[GSP Batch] --> G
    E3 --> F3[NWP Batch] --> G
    E4 --> F4[Satelite Batch] --> G
    G[Example]

```


## PVn and NWP Pipeline

`pv_nwp.py` is a training pipeline for loading NWP and PV data.

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


## PV, Satellite and NWP Pipeline

`pv_satellite_nwp.py` is a training pipeline for loading NWP,PV, and Satellite.

The location is chosen using the center of the GSP location.
Then a time is chosen, and PV, Satellite and NWP examples are made.

```mermaid
graph TD
    A1([Load PV])
    A2([Load NWP])
    B[Select Location]
    A3([Load Satellite])
    A1 --> C1
    A2 --> C2
    A1 --> B
    C1[PV location]
    C2[NWP location]
    C3[Satellite location]
    B --> C1 --> D1
    B --> C2 --> D1
    B --> C3 --> D1
    A3 --> C3
    D1[Select Joint Time Periods] --> D2
    D2[Select T0]
    C1 --> E1
    C2 --> E2
    D2 --> E1[PV Time and location]
    D2 --> E2[NWP Time and location]
    D2 --> E3[Satellite Time and location]
    C3 --> E3
    E1 --> F1[PV Batch] --> G
    E2 --> F2[NWP Batch] --> G
    E3 --> F3[Satelite Batch] --> G
    G[Example]

```
