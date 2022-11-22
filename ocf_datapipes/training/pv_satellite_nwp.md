# GSP PV Satellite and NWP Pipeline

gsp_pv_satellite_nwp.py is a training pipeline for loading NWP,PV, and Satellite.

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
