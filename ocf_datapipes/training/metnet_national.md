# MetNet National Pipeline

metnet_national.py is a training pipeline for loading NWP,PV,Satellite,and Topographic data and transforming it as in the MetNet paper.

The location is chosen using the center of the National GSP shape. Only the modalities wanted are loaded.
Then a time is chosen, and PV and NWP examples are made.

```mermaid
graph TD
    A[Load GSP] -->|Select Train/Test Times| B(Drop Regional GSP) --> A1
    C[Load NWP] --> CA[Filter] --> A1
    D[Load Satellite] --> DA[Filter] --> A1
    E[Load PV] --> EA[Filter] --> A1
    F[Load Topo]
    A1[Select Joint Time Periods]
    B1[Select T0 Time]
    A1 --> B1
    B1 --> C1
    A1 --> C1
    B1 --> CAA
    CA --> CAA[Convert to Target Time]
    DA --> C1
    EA --> C1
    C1[Select Time Slice]
    AA[Get Location]
    B --> AA
    A11[PreProcess MetNet]
    C1 --> A11
    CAA --> A11
    F --> A11
    AA --> A11
    A111[Return Example]
    A11 --> A111
```
