""" The idea is visualise one of the batches

This is a bit of a work in progress, but the idea is to visualise the batch in a markdown file.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import torch

from ocf_datapipes.batch import BatchKey, NumpyBatch, NWPBatchKey


def visualise_batch(batch: NumpyBatch, folder=".", output_file="report.md", limit_examples=None):
    """Visualize the batch in a markdown file"""

    # create dir if it does not exist
    for d in [folder, f"{folder}/gsp", f"{folder}/nwp", f"{folder}/satellite"]:
        if not os.path.exists(d):
            os.makedirs(d)

    with open(f"{folder}/{output_file}", "a") as f:
        # Wind
        print("# Batch visualisation", file=f)

        print("## Wind \n", file=f)
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
                print("\n", file=f)
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    print(f"{key} {value.shape=}", file=f)
                    print(f"Max {value.max()}", file=f)
                    print(f"Min {value.min()}", file=f)
                elif isinstance(value, int):
                    print(f"{key} {value}", file=f)
                else:
                    print(f"{key} {value}", file=f)

        print("## GSP \n", file=f)
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
                print("\n", file=f)
                print(f"### {key.name}", file=f)
                value = batch[key]
                if key.name == "gsp":
                    # plot gsp data
                    n_examples = value.shape[0]
                    if limit_examples is not None:
                        n_examples = min(n_examples, limit_examples)

                    for b in range(n_examples):
                        fig = go.Figure()
                        gsp_data = value[b, :, 0]
                        time = pd.to_datetime(batch[BatchKey.gsp_time_utc][b], unit="s")
                        fig.add_trace(go.Scatter(x=time, y=gsp_data, mode="lines", name="GSP"))
                        fig.update_layout(
                            title=f"GSP - example {b}", xaxis_title="Time", yaxis_title="Value"
                        )
                        # fig.show(renderer='browser')
                        name = f"gsp/gsp_{b}.png"
                        fig.write_image(f"{folder}/{name}")
                        print(f"![](./{name})", file=f)
                        print("\n", file=f)

                elif isinstance(value, torch.Tensor):
                    print(f"shape {value.shape=}", file=f)
                    print(f"Max {value.max():.2f}", file=f)
                    print(f"Min {value.min():.2f}", file=f)
                elif isinstance(value, int):
                    print(f"{value}", file=f)
                else:
                    print(f"{value}", file=f)

                # TODO plot solar azimuth and elevation

        # NWP
        print("## NWP \n", file=f)

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
            print("\n", file=f)
            print(f"### Provider {provider}", file=f)
            nwp_provider = nwp[provider]

            # plot nwp main data
            nwp_data = nwp_provider[NWPBatchKey.nwp]
            # average of lat and lon
            nwp_data = nwp_data.mean(dim=(3, 4))

            n_examples = nwp_data.shape[0]
            if limit_examples is not None:
                n_examples = min(n_examples, limit_examples)

            for b in range(n_examples):

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
                name = f"nwp/{provider}_nwp_{b}.png"
                fig.write_image(f"{folder}/{name}")
                print(f"![](./{name})", file=f)
                print("\n", file=f)

            for key in keys:
                print("\n", file=f)
                print(f"#### {key.name}", file=f)
                value = nwp_provider[key]

                if "time" in key.name:

                    # make a table with example, shape, max, min
                    print("| Example | Shape | Max | Min |", file=f)
                    print("| --- | --- | --- | --- |", file=f)

                    for example_id in range(n_examples):
                        value_ts = pd.to_datetime(value[example_id], unit="s")
                        print(
                            f"| {example_id} | {len(value_ts)} "
                            f"| {value_ts.max()} | {value_ts.min()} |",
                            file=f,
                        )

                elif "channel" in key.name:

                    # create a table with the channel names with max, min, mean and std
                    print("| Channel | Max | Min | Mean | Std |", file=f)
                    print("| --- | --- | --- | --- | --- |", file=f)
                    for i in range(len(value)):
                        channel = value[i]
                        data = nwp_data[:, :, i]
                        print(
                            f"| {channel} "
                            f"| {data.max().item():.2f} "
                            f"| {data.min().item():.2f} "
                            f"| {data.mean().item():.2f} "
                            f"| {data.std().item():.2f} |",
                            file=f,
                        )

                    print(f"Shape={value.shape}", file=f)

                elif isinstance(value, torch.Tensor):
                    print(f"Shape {value.shape=}", file=f)
                    print(f"Max {value.max():.2f}", file=f)
                    print(f"Min {value.min():.2f}", file=f)
                elif isinstance(value, int):
                    print(f"{value}", file=f)
                else:
                    print(f"{value}", file=f)

        # Satellite
        print("## Satellite \n", file=f)
        keys = [
            BatchKey.satellite_actual,
            BatchKey.satellite_t0_idx,
            BatchKey.satellite_time_utc,
            BatchKey.satellite_time_utc,
            BatchKey.satellite_x_geostationary,
            BatchKey.satellite_y_geostationary,
        ]

        for key in keys:

            print("\n", file=f)
            print(f"#### {key.name}", file=f)
            value = batch[key]

            if "satellite_actual" in key.name:

                print(value.shape, file=f)

                # average of lat and lon
                value = value.mean(dim=(3, 4))

                n_examples = value.shape[0]
                if limit_examples is not None:
                    n_examples = min(n_examples, limit_examples)

                for b in range(n_examples):

                    fig = go.Figure()
                    for i in range(value.shape[2]):
                        satellite_data_one_channel = value[b, :, i]
                        time = batch[BatchKey.satellite_time_utc][b]
                        time = pd.to_datetime(time, unit="s")
                        fig.add_trace(
                            go.Scatter(x=time, y=satellite_data_one_channel, mode="lines")
                        )

                    fig.update_layout(
                        title=f"Satellite - example {b}", xaxis_title="Time", yaxis_title="Value"
                    )
                    # fig.show(renderer='browser')
                    name = f"satellite/satellite_{b}.png"
                    fig.write_image(f"{folder}/{name}")
                    print(f"![](./{name})", file=f)
                    print("\n", file=f)

            elif "time" in key.name:

                # make a table with example, shape, max, min
                print("| Example | Shape | Max | Min |", file=f)
                print("| --- | --- | --- | --- |", file=f)

                for example_id in range(n_examples):
                    value_ts = pd.to_datetime(value[example_id], unit="s")
                    print(
                        f"| {example_id} | {len(value_ts)} "
                        f"| {value_ts.max()} | {value_ts.min()} |",
                        file=f,
                    )

            elif "channel" in key.name:

                # create a table with the channel names with max, min, mean and std
                print("| Channel | Max | Min | Mean | Std |", file=f)
                print("| --- | --- | --- | --- | --- |", file=f)
                for i in range(len(value)):
                    channel = value[i]
                    data = nwp_data[:, :, i]
                    print(
                        f"| {channel} "
                        f"| {data.max().item():.2f} "
                        f"| {data.min().item():.2f} "
                        f"| {data.mean().item():.2f} "
                        f"| {data.std().item():.2f} |",
                        file=f,
                    )

                print(f"Shape={value.shape}", file=f)

            elif isinstance(value, torch.Tensor):
                print(f"Shape {value.shape=}", file=f)
                print(f"Max {value.max():.2f}", file=f)
                print(f"Min {value.min():.2f}", file=f)
            elif isinstance(value, int):
                print(f"{value}", file=f)
            else:
                print(f"{value}", file=f)


# For example you can run it like this
# with open("batch.md", "w") as f:
#     sys.stdout = f
#     d = torch.load("000000.pt")
#     visualise_batch(d)
