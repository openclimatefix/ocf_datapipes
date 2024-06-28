""" The idea is visualize one of the batches

This is a bit of a working progress, but the idea is to visualize the batch in a markdown file.
"""
import pandas as pd
import sys

from ocf_datapipes.batch import NumpyBatch, BatchKey, NWPBatchKey
import torch
import plotly.graph_objects as go


def visualize_batch(batch: NumpyBatch, example_id: int = 0):

    # Wind
    print("# Batch visualization")

    print(f"We are looking at example {example_id}")

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
            if isinstance(value, torch.Tensor):
                print(f"shape {value.shape=}")
                print(f"Max {value.max():.2f}")
                print(f"Min {value.min():.2f}")
            elif isinstance(value, int):
                print(f"{value}")
            else:
                print(f"{value}")

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
        fig = go.Figure()
        for i in range(len(nwp_provider[NWPBatchKey.nwp_channel_names])):
            channel = nwp_provider[NWPBatchKey.nwp_channel_names][i]
            nwp_data_one_channel = nwp_data[example_id, :, i]
            time = nwp_provider[NWPBatchKey.nwp_target_time_utc][example_id]
            time = pd.to_datetime(time, unit="s")
            fig.add_trace(go.Scatter(x=time, y=nwp_data_one_channel, mode="lines", name=channel))

        fig.update_layout(title=f"{provider} NWP", xaxis_title="Time", yaxis_title="Value")
        # fig.show(renderer='browser')
        name = f"{provider}_nwp.png"
        fig.write_image(name)
        print(f"![]({name})")
        print("\n")

        for key in keys:
            print("\n")
            print(f"#### {key.name}")
            value = nwp_provider[key]

            if "time" in key.name:
                value = pd.to_datetime(value[example_id], unit="s")
                print(f"Shape={value.shape}")
                print(f"Max {value.max()}")
                print(f"Min {value.min()}")

            elif "channel" in key.name:

                # create a table with the channel names with max, min, mean and std
                print("| Channel | Max | Min | Mean | Std |")
                print("| --- | --- | --- | --- | --- |")
                for i in range(len(value)):
                    channel = value[i]
                    data = nwp_data[:, :, i]
                    print(
                        f"| {channel} | {data.max().item():.2f} | {data.min().item():.2f} | {data.mean().item():.2f} | {data.std().item():.2f} |"
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
#     visualize_batch(d, example_id=3)
