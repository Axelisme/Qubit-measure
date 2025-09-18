import numpy as np
import plotly.graph_objects as go

from zcu_tools.simulate import mA2flx

from ..models import energy2transition
from ..processing import cast2real_and_norm


class VisualizeSpect:
    def __init__(
        self, s_spects, s_mAs, s_fpts, mAs, energies, allows, auto_hide=False
    ) -> None:
        self.s_spects = s_spects
        self.s_mAs = s_mAs
        self.s_fpts = s_fpts
        self.mAs = mAs
        self.energies = energies
        self.allows = allows

        self.auto_hide = auto_hide  # 新增參數，預設為 False

        # Default scatter point styling
        self.scatter_size = 3
        self.scatter_color = "red"
        self.scatter_color_array = None  # 用於存儲顏色陣列

        if len(self.s_spects) == 0:
            raise ValueError("No spectrum data provided")

        first_spect = next(iter(self.s_spects.values()))
        mA_c, period = first_spect["mA_c"], first_spect["period"]
        self.s_flxs = mA2flx(self.s_mAs, mA_c, period)
        self.flxs = mA2flx(self.mAs, mA_c, period)

        self.mA_c = mA_c
        self.period = period

    def set_scatter_style(self, size=None, color=None) -> "VisualizeSpect":
        if size is not None:
            self.scatter_size = size
        if color is not None:
            # 可以是顏色名稱或數值陣列
            self.scatter_color = color
            # 檢查是否為陣列
            if isinstance(color, (list, np.ndarray)):
                self.scatter_color_array = color

        return self  # For method chaining

    def plot_background(self, fig) -> None:
        # Add heatmap traces for each spectrum in s_spects
        for spect in self.s_spects.values():
            # Get corresponding data and range
            signals = spect["spectrum"]["data"] ** 1.5
            flx_mask = np.any(~np.isnan(signals), axis=0)
            fpt_mask = np.any(~np.isnan(signals), axis=1)
            signals = signals[fpt_mask, :][:, flx_mask]

            # Normalize data
            amps = cast2real_and_norm(signals)

            # Add heatmap trace
            fig.add_trace(
                go.Heatmap(
                    z=amps,
                    x=spect["spectrum"]["mAs"][flx_mask],
                    y=spect["spectrum"]["fpts"][fpt_mask],
                    colorscale="Greys",
                    showscale=False,
                )
            )

    def plot_predict_lines(self, fig) -> None:
        # Calculate transitions
        fs, labels = energy2transition(self.energies, self.allows)

        # 計算哪些線需要隱藏
        visible_lines = self._filter_nearby_lines(fs, self.mAs, self.s_fpts, self.s_mAs)

        # Add transition line traces
        for i, label in enumerate(labels):
            visible = "legendonly" if not visible_lines[i] else True
            fig.add_trace(
                go.Scatter(
                    x=self.mAs,
                    y=fs[:, i],
                    mode="lines",
                    name=label,
                    visible=visible,
                )
            )

    def plot_scatter_point(self, fig) -> None:
        marker_dict = {"size": self.scatter_size}

        # 處理顏色設置
        if self.scatter_color_array is not None:
            # 如果提供了顏色陣列，使用它並指定顏色範圍
            marker_dict["color"] = self.scatter_color_array
            marker_dict["colorscale"] = "Viridis"
            marker_dict["showscale"] = True  # 顯示顏色刻度
            marker_dict["colorbar"] = dict(
                x=-0.1,  # 將顏色條放置在左側
                xanchor="left",  # 錨點在左側
            )
            hovertext = self.scatter_color_array
        else:
            # 否則使用單一顏色
            marker_dict["color"] = self.scatter_color
            hovertext = None

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=self.s_mAs,
                y=self.s_fpts,
                mode="markers",
                marker=marker_dict,
                hovertext=hovertext,
            )
        )

    def plot_constant_freqs(self, fig) -> None:
        if "r_f" in self.allows:
            fig.add_hline(y=self.allows["r_f"], line_dash="dash", name="r_f")

        if "sample_f" in self.allows:
            fig.add_hline(y=self.allows["sample_f"], line_dash="dash", name="sample_f")

    def _set_axis_limit(self, fig) -> None:
        sp_mAs = np.concatenate([s["spectrum"]["mAs"] for s in self.s_spects.values()])
        sp_fpts = np.concatenate(
            [s["spectrum"]["fpts"] for s in self.s_spects.values()]
        )
        mA_bound = (
            min(np.nanmin(sp_mAs), self.mAs.min()),
            max(np.nanmax(sp_mAs), self.mAs.max()),
        )
        fpt_bound = (
            min(
                np.nanmin(sp_fpts),
                self.allows.get("r_f", np.inf) - 0.1,
                self.allows.get("sample_f", np.inf) - 0.1,
            ),
            max(
                np.nanmax(sp_fpts),
                self.allows.get("r_f", 0.0) + 0.1,
                self.allows.get("sample_f", 0.0) + 0.1,
            ),
        )

        if len(self.s_fpts) > 0:
            mA_bound = (
                min(mA_bound[0], self.s_mAs.min()),
                max(mA_bound[1], self.s_mAs.max()),
            )
            fpt_bound = (
                min(fpt_bound[0], self.s_fpts.min()),
                max(fpt_bound[1], self.s_fpts.max()),
            )

        # Set x and y axis range
        fig.update_xaxes(range=[mA_bound[0], mA_bound[1]])
        fig.update_yaxes(range=[0.0, fpt_bound[1]])

    def create_figure(self) -> go.Figure:
        fig = go.Figure()

        self.plot_background(fig)
        self.plot_predict_lines(fig)
        self.plot_constant_freqs(fig)
        self.plot_scatter_point(fig)

        self._set_axis_limit(fig)

        # Secondary x axis, show flxs
        fig.add_scatter(
            x=self.flxs, y=np.zeros_like(self.flxs), xaxis="x2", opacity=0.0
        )
        ticks_mAs = self.mAs[:: max(1, len(self.mAs) // 20)]
        ticks_flxs = self.flxs[:: max(1, len(self.flxs) // 20)]
        fig.update_layout(
            xaxis2=dict(
                tickvals=ticks_mAs,
                ticktext=[f"{flx:.2f}" for flx in ticks_flxs],
                matches="x1",
                overlaying="x1",
                side="top",
            )
        )

        # Update layout
        fig.update_layout(
            margin=dict(t=100),
            legend_title_text="Transition",
            title_x=0.5,
            xaxis_title="mAs",
            yaxis_title="Frequency (GHz)",
            legend=dict(x=1, y=0.5),
            height=1600,
        )

        return fig

    def _filter_nearby_lines(self, fs, mAs, s_fpts, s_mAs) -> np.ndarray:
        """
        計算哪些轉換線靠近散點，並返回布林陣列，決定要顯示哪些線。

        Parameters:
        fs: numpy array, 所有轉換線的頻率數據 (M, K)
        mAs: numpy array, 所有轉換線對應的通量數據 (M, )
        s_fpts: numpy array, 所有散點的頻率數據 (N, )
        s_mAs: numpy array, 所有散點的通量數據 (N, )

        Returns:
        visible_lines: numpy array, 形狀 (K, ), True 表示該線要顯示, False 表示要隱藏
        """
        K = fs.shape[1]

        THRESHOLD = 4

        if self.auto_hide:
            # interpolate flux points
            s_fs = np.array(
                [np.interp(s_mAs, mAs, fs[:, i]) for i in range(fs.shape[1])]
            ).T  # (N, K)

            # 計算散點與所有線之間的距離
            dists = np.abs(s_fs - s_fpts[:, None])  # (N, K)
            matchs = np.argmin(dists, axis=1)  # (N, )

            # if only one or two points are matched this line, make it invisible
            visible_lines = (
                np.sum(matchs[:, None] == np.arange(K)[None, :], axis=0) > THRESHOLD
            )
        else:
            visible_lines = np.full(K, True)

        return visible_lines
