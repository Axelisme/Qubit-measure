import numpy as np
import pandas as pd
import plotly.express as px
import scqubits as scq
from tqdm.auto import tqdm


def scan_params_space(EJb, EC, ELb, avoid_freqs, Temp, noise_channels):
    t1s = []
    m01s = []
    paramss = []
    collisions = []

    scq.settings.T1_DEFAULT_WARNING = False
    fluxonium = scq.Fluxonium(EJb[0], EC, ELb[0], flux=0.5, cutoff=40, truncated_dim=7)
    for EJ in tqdm(np.arange(EJb[0], EJb[1], 0.05), desc="EJ"):
        for EL in np.arange(ELb[0], ELb[1], 0.05):
            fluxonium.EJ = EJ
            fluxonium.EC = EC
            fluxonium.EL = EL

            spectData = fluxonium.eigensys(evals_count=7, return_spectrumdata=True)
            eval_eng, eval_sys = spectData.energy_table, spectData.state_table
            e0x = eval_eng - eval_eng[0]
            e1x = eval_eng - eval_eng[1]

            freqs = np.array(avoid_freqs)[None, :]
            all_collision = np.all(
                np.logical_or(
                    np.any(np.abs(e0x[:, None] - freqs) < 0.3, axis=0),
                    np.any(np.abs(e1x[:, None] - freqs) < 0.3, axis=0),
                )
            )

            e01 = eval_eng[1] - eval_eng[0]
            for i in range(eval_eng.shape[0]):
                for j in range(i + 1, eval_eng.shape[0]):
                    eij = eval_eng[j] - eval_eng[i]
                    if (i != 0 or j != 1) and np.abs(eij - e01) < 0.3:
                        all_collision = True
                        break
                if all_collision:
                    break

            elements = fluxonium.n_operator(energy_esys=(eval_eng, eval_sys))
            t1_eff = fluxonium.t1_effective(
                esys=(eval_eng, eval_sys),
                noise_channels=noise_channels,
                common_noise_options=dict(i=1, j=0, T=Temp),
            )

            t1s.append(t1_eff)
            m01s.append(elements[0, 1])
            paramss.append((EJ, EC, EL))
            collisions.append(all_collision)
    t1s = np.array(t1s)
    m01s = np.array(m01s)
    paramss = np.array(paramss)
    collisions = np.array(collisions)

    return pd.DataFrame(
        {
            "EJ": paramss[:, 0],
            "EC": paramss[:, 1],
            "EL": paramss[:, 2],
            "t1": t1s,
            "m01": np.abs(m01s),
            "collision": collisions,
        }
    )


def plot_scan_results(data):
    data["Label"] = data.apply(
        lambda row: f"EJ={row['EJ']:.2f}, EC={row['EC']:.3f}, EL={row['EL']:.3f}",
        axis=1,
    )

    # 繪製散點圖
    fig = px.scatter(
        data,
        x="m01",
        y="t1",
        color="collision",
        color_discrete_map={True: "red", False: "blue"},
        log_x=True,
        log_y=True,
        hover_name="Label",
        labels={"Matrix Element 0-1": "m01", "T1 (us)": "t1"},
    )
    fig.update_traces(marker=dict(size=5))

    max_id = np.argmax(data["t1"].where(~data["collision"], 0.0))

    # Add annotation for the max_id point
    max_point = data.iloc[max_id]
    fig.add_annotation(
        x=np.log10(max_point["m01"]),
        y=np.log10(max_point["t1"]),
        text=max_point["Label"],
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        xref="x",
        yref="y",
    )

    fig.update_layout(
        xaxis_title="Matrix Element 0-1",
        yaxis_title="T1 (us)",
        xaxis=dict(tickformat=".0e", type="log"),  # Ensure axis type is log
        yaxis=dict(tickformat=".0e", type="log"),  # Ensure axis type is log
        title_x=0.501,
        template="plotly_white",
        showlegend=True,
        width=1100,
        height=750,
    )

    return fig, (float(max_point["EJ"]), float(max_point["EC"]), float(max_point["EL"]))
