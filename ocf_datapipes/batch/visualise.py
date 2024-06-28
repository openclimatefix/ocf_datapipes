""" The idea is visualize one of the batches

This is a bit of a working progress, but the idea is to visualize the batch in a markdown file.
"""

import pandas as pd
import plotly.graph_objects as go
import torch

from ocf_datapipes.batch import BatchKey, NumpyBatch, NWPBatchKey


def visualize_batch(batch: NumpyBatch):
    """Visualize the batch in a markdown file"""

    # Wind
    print("# Batch visualization")

    print("## Wind \n")
    keys = [
        BatchKey.wind,
        BatchKey.wind_t0_idx,
        BatchKey.wind_time_utc,
        BatchKey.wind_id,
        BatchKey.wind_observed_capacity_mwp,
        BatchKey.wind_nominal_capacity_mwp,
        BatchKey.wind_time_utc,
        BatchKey.wind_latitude,
        BatchKey.wind_longitude,
        BatchKey.wind_solar_azimuth,
        BatchKey.wind_solar_elevation,
    ]
    for key in keys:
        if key in batch.keys():
            print("\n")
            value = batch[key]
            if isinstance(value, torch.Tensor):
                print(f"{key} {value.shape=}")
                print(f"Max {value.max()}")
                print(f"Min {value.min()}")
            elif isinstance(value, int):
                print(f"{key} {value}")
            else:
                print(f"{key} {value}")

    print("## GSP \n")
    keys = [
        BatchKey.gsp,
        BatchKey.gsp_id,
        BatchKey.gsp_time_utc,
        BatchKey.gsp_time_utc_fourier,
        BatchKey.gsp_x_osgb,
        BatchKey.gsp_x_osgb_fourier,
        BatchKey.gsp_y_osgb,
        BatchKey.gsp_y_osgb_fourier,
        BatchKey.gsp_t0_idx,
        BatchKey.gsp_effective_capacity_mwp,
        BatchKey.gsp_nominal_capacity_mwp,
        BatchKey.gsp_solar_azimuth,
        BatchKey.gsp_solar_elevation,
    ]
    for key in keys:
        if key in batch.keys():
            print("\n")
            print(f"### {key.name}")
            value = batch[key]
            if key.name == "gsp":
                # plot gsp data
                for b in range(value.shape[0]):
                    fig = go.Figure()
                    gsp_data = value[b, :, 0]
                    time = pd.to_datetime(batch[BatchKey.gsp_time_utc][b], unit="s")
                    fig.add_trace(go.Scatter(x=time, y=gsp_data, mode="lines", name="GSP"))
                    fig.update_layout(
                        title=f"GSP - example {b}", xaxis_title="Time", yaxis_title="Value"
                    )
                    # fig.show(renderer='browser')
                    name = f"gsp_{b}.png"
                    fig.write_image(name)
                    print(f"![]({name})")
                    print("\n")

            elif isinstance(value, torch.Tensor):
                print(f"shape {value.shape=}")
                print(f"Max {value.max():.2f}")
                print(f"Min {value.min():.2f}")
            elif isinstance(value, int):
                print(f"{value}")
            else:
                print(f"{value}")

            # TODO plot solar azimuth and elevation

    # NWP
    print("## NWP \n")

    keys = [
        NWPBatchKey.nwp,
        NWPBatchKey.nwp_target_time_utc,
        NWPBatchKey.nwp_channel_names,
        NWPBatchKey.nwp_step,
        NWPBatchKey.nwp_t0_idx,
        NWPBatchKey.nwp_init_time_utc,
    ]

    nwp = batch[BatchKey.nwp]

    nwp_providers = nwp.keys()
    for provider in nwp_providers:
        print("\n")
        print(f"### Provider {provider}")
        nwp_provider = nwp[provider]

        # plot nwp main data
        nwp_data = nwp_provider[NWPBatchKey.nwp]
        # average of lat and lon
        nwp_data = nwp_data.mean(dim=(3, 4))

        for b in range(nwp_data.shape[0]):

            fig = go.Figure()
            for i in range(len(nwp_provider[NWPBatchKey.nwp_channel_names])):
                channel = nwp_provider[NWPBatchKey.nwp_channel_names][i]
                nwp_data_one_channel = nwp_data[b, :, i]
                time = nwp_provider[NWPBatchKey.nwp_target_time_utc][b]
                time = pd.to_datetime(time, unit="s")
                fig.add_trace(
                    go.Scatter(x=time, y=nwp_data_one_channel, mode="lines", name=channel)
                )

            fig.update_layout(
                title=f"{provider} NWP - example {b}", xaxis_title="Time", yaxis_title="Value"
            )
            # fig.show(renderer='browser')
            name = f"{provider}_nwp_{b}.png"
            fig.write_image(name)
            print(f"![]({name})")
            print("\n")

        for key in keys:
            print("\n")
            print(f"#### {key.name}")
            value = nwp_provider[key]

            if "time" in key.name:

                # make a table with example, shape, max, min
                print("| Example | Shape | Max | Min |")
                print("| --- | --- | --- | --- |")

                for example_id in range(value.shape[0]):
                    value_ts = pd.to_datetime(value[example_id], unit="s")
                    print(
                        f"| {example_id} | {len(value_ts)} | {value_ts.max()} | {value_ts.min()} |"
                    )

            elif "channel" in key.name:

                # create a table with the channel names with max, min, mean and std
                print("| Channel | Max | Min | Mean | Std |")
                print("| --- | --- | --- | --- | --- |")
                for i in range(len(value)):
                    channel = value[i]
                    data = nwp_data[:, :, i]
                    print(
                        f"| {channel} "
                        f"| {data.max().item():.2f} "
                        f"| {data.min().item():.2f} "
                        f"| {data.mean().item():.2f} "
                        f"| {data.std().item():.2f} |"
                    )

                print(f"Shape={value.shape}")

            elif isinstance(value, torch.Tensor):
                print(f"Shape {value.shape=}")
                print(f"Max {value.max():.2f}")
                print(f"Min {value.min():.2f}")
            elif isinstance(value, int):
                print(f"{value}")
            else:
                print(f"{value}")


# For example you can run it like this
# with open("batch.md", "w") as f:
#     sys.stdout = f
#     d = torch.load("000000.pt")
#     visualize_batch(d)
