import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Forzar backend interactivo en Linux
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
from rtlsdr import RtlSdr
from scipy.signal import sosfilt_zi

from src.config import (
    ALPHA,
    BP_BW_KHZ_MAX,
    BP_BW_KHZ_MIN,
    BP_OFFSET_KHZ_MAX,
    FFT_SIZE,
    FREQ_MAX_MHZ,
    FREQ_MIN_MHZ,
    PSD_BUFFER_SIZE,
    PSD_UPDATE_HZ,
    SAMPLE_RATE_HZ,
    SPAN_MHZ_MAX,
    SPAN_MHZ_MIN,
)
from src.backend import (
    IQRingBuffer,
    PaplayAudioPlayer,
    SDRPipeline,
    SounddeviceAudioPlayer,
    apply_bandpass,
    design_lowpass_sos,
    freq_axis_mhz,
    make_test_beep,
    measure_signal,
    parse_float,
    passband_mask_rf,
    welch_psd_db,
)


def run_gui(
    sdr: RtlSdr,
    spectrum_ring: IQRingBuffer,
    pipeline: SDRPipeline,
    audio_player: PaplayAudioPlayer | SounddeviceAudioPlayer | None,
    state: dict,
) -> None:
    """Interfaz gráfica Matplotlib y bucle principal de actualización del espectro."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.patch.set_facecolor("#fafbfc")

    line, = ax.plot(freq_axis_mhz(state["center_freq_hz"]), np.zeros(FFT_SIZE), label="Espectro")
    line_filt, = ax.plot(
        freq_axis_mhz(state["center_freq_hz"]),
        np.zeros(FFT_SIZE),
        color="tab:green",
        linewidth=2.0,
        zorder=3,
        label="Filtrado",
    )
    state["bp_span"] = ax.axvspan(0, 0, alpha=0.15, color="tab:green", label="Pasabanda")
    metrics_text = ax.text(
        0.02, 0.97, "",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )
    ax.legend(loc="upper right")
    ax.set_title(
        f"PSD — RTL: {state['center_freq_hz'] / 1e6:.3f} MHz | Span: {state['span_mhz']:.3f} MHz"
    )
    ax.set_xlabel("Frecuencia [MHz]")
    ax.set_ylabel("PSD [dB/Hz]")
    ax.grid(True)

    def update_spectrum_xlim() -> None:
        center_mhz = state["center_freq_hz"] / 1e6
        half = state["span_mhz"] / 2.0
        ax.set_xlim(center_mhz - half, center_mhz + half)

    update_spectrum_xlim()
    ax.set_ylim(-150, -20)

    def update_bp_span() -> None:
        state["bp_span"].remove()
        rf_c = state["center_freq_hz"] + state["bp_center_offset_hz"]
        half = state["bp_bandwidth_hz"] / 2
        lo_mhz = (rf_c - half) / 1e6
        hi_mhz = (rf_c + half) / 1e6
        state["bp_span"] = ax.axvspan(lo_mhz, hi_mhz, alpha=0.15, color="tab:green")

    def redesign_bandpass(*, reset_state: bool = True) -> None:
        pipeline.update_filter(
            state["bp_center_offset_hz"],
            state["bp_bandwidth_hz"],
            reset=reset_state,
        )
        update_bp_span()
        state["psd_filt_db_avg"] = None

    def set_bp_center_offset_khz(val_khz: float, *, update_box: bool = True) -> None:
        state["bp_center_offset_hz"] = float(np.clip(val_khz, -BP_OFFSET_KHZ_MAX, BP_OFFSET_KHZ_MAX)) * 1e3
        if update_box:
            bp_center_box.set_val(f"{val_khz:.1f}")
        redesign_bandpass()
        fig.canvas.draw_idle()

    def set_bp_bandwidth_khz(val_khz: float, *, update_box: bool = True) -> None:
        state["bp_bandwidth_hz"] = float(np.clip(val_khz, BP_BW_KHZ_MIN, BP_BW_KHZ_MAX)) * 1e3
        if update_box:
            bp_bw_box.set_val(f"{val_khz:.1f}")
        redesign_bandpass()
        fig.canvas.draw_idle()

    def set_span_mhz(val_mhz: float, *, update_box: bool = True) -> None:
        state["span_mhz"] = float(np.clip(val_mhz, SPAN_MHZ_MIN, SPAN_MHZ_MAX))
        if update_box:
            span_box.set_val(f"{state['span_mhz']:.3f}")
        update_spectrum_xlim()
        fig.canvas.draw_idle()

    def format_metrics(m: dict) -> str:
        if m is None:
            return "Filtro: sin métricas"
        fc_mhz = m["fc_hz"] / 1e6
        bw_khz = m["bw_hz"] / 1e3
        p_db = m["power_db"]
        p_str = f"{p_db:.1f} dB" if np.isfinite(p_db) else "—"
        audio_str = ""
        if audio_player is not None and audio_player.enabled:
            pk = getattr(audio_player, "_peak", 0.0)
            audio_str = f"\nAudio pico: {pk:.3f}"
        demod = state.get("demod_mode", "—")
        return (
            f"Fc señal: {fc_mhz:.6f} MHz\n"
            f"BW (−3 dB): {bw_khz:.3f} kHz\n"
            f"Span vista: {state['span_mhz']:.3f} MHz\n"
            f"Potencia: {p_str}\n"
            f"Demod: {demod}{audio_str}"
        )

    redesign_bandpass()

    def set_center_freq_mhz(val_mhz: float, *, update_box: bool = True) -> None:
        val_mhz = float(np.clip(val_mhz, FREQ_MIN_MHZ, FREQ_MAX_MHZ))
        state["center_freq_hz"] = val_mhz * 1e6
        sdr.center_freq = state["center_freq_hz"]

        ax.set_title(f"PSD — RTL: {val_mhz:.3f} MHz | Span: {state['span_mhz']:.3f} MHz")
        state["psd_db_avg"] = None
        state["psd_filt_db_avg"] = None
        update_spectrum_xlim()
        update_bp_span()

        if update_box:
            rtl_box.set_val(f"{val_mhz:.3f}")
        fig.canvas.draw_idle()

    def auto_tune_to_peak() -> None:
        """Centra el filtro en el pico del espectro visible."""
        if state["last_freqs"] is None or state["psd_db_avg"] is None:
            print("Auto-sintonía: espera a que aparezca el espectro.")
            return
        peak_idx = int(np.argmax(state["psd_db_avg"]))
        offset_khz = float(state["last_freqs"][peak_idx] / 1e3)
        set_bp_center_offset_khz(offset_khz, update_box=True)
        if state["demod_mode"] == "FM" and state["bp_bandwidth_hz"] < 150_000:
            set_bp_bandwidth_khz(250.0, update_box=True)
        print(f"Auto-sintonía: centro rel. → {offset_khz:.1f} kHz")

    # --- Panel de controles (rejilla ordenada) ---
    CTRL = dict(
        left=0.06,
        right=0.97,
        gap=0.014,
        box_h=0.048,
        row_params=0.155,
        row_actions=0.055,
        btn_h=0.044,
        label_fs=8.5,
        section_fs=9.0,
    )
    n_cols = 5
    total_w = CTRL["right"] - CTRL["left"]
    col_w = (total_w - (n_cols - 1) * CTRL["gap"]) / n_cols

    def col_left(index: int) -> float:
        return CTRL["left"] + index * (col_w + CTRL["gap"])

    def style_field_ax(ax) -> None:
        ax.set_facecolor("#f4f6f8")
        for spine in ax.spines.values():
            spine.set_color("#b0b8c4")
            spine.set_linewidth(0.8)

    def add_param_field(col: int, title: str, initial: str, on_submit) -> TextBox:
        left = col_left(col)
        fig.text(
            left + col_w / 2,
            CTRL["row_params"] + CTRL["box_h"] + 0.012,
            title,
            ha="center",
            va="bottom",
            fontsize=CTRL["label_fs"],
            color="#2c3e50",
        )
        ax = fig.add_axes([left, CTRL["row_params"], col_w, CTRL["box_h"]])
        style_field_ax(ax)
        box = TextBox(ax, "", initial=initial, textalignment="center")
        box.on_submit(on_submit)
        return box

    fig.text(
        CTRL["left"],
        CTRL["row_params"] + CTRL["box_h"] + 0.038,
        "Parámetros  ·  Enter para aplicar",
        fontsize=CTRL["section_fs"],
        color="#5a6a7a",
        fontweight="bold",
    )

    def on_rtl_text(text: str) -> None:
        val = parse_float(text)
        if val is None:
            rtl_box.set_val(f"{state['center_freq_hz'] / 1e6:.3f}")
            return
        set_center_freq_mhz(val, update_box=False)

    rtl_box = add_param_field(
        0, "RTL central (MHz)", f"{state['center_freq_hz'] / 1e6:.3f}", on_rtl_text
    )

    def on_bp_center_text(text: str) -> None:
        val = parse_float(text)
        if val is None:
            bp_center_box.set_val(f"{state['bp_center_offset_hz'] / 1e3:.1f}")
            return
        set_bp_center_offset_khz(val, update_box=False)

    bp_center_box = add_param_field(
        1, "Δf filtro (kHz)", f"{state['bp_center_offset_hz'] / 1e3:.1f}", on_bp_center_text
    )

    def on_bp_bw_text(text: str) -> None:
        val = parse_float(text)
        if val is None:
            bp_bw_box.set_val(f"{state['bp_bandwidth_hz'] / 1e3:.1f}")
            return
        set_bp_bandwidth_khz(val, update_box=False)

    bp_bw_box = add_param_field(
        2, "Ancho BW (kHz)", f"{state['bp_bandwidth_hz'] / 1e3:.1f}", on_bp_bw_text
    )

    def on_span_text(text: str) -> None:
        val = parse_float(text)
        if val is None:
            span_box.set_val(f"{state['span_mhz']:.3f}")
            return
        set_span_mhz(val, update_box=False)

    span_box = add_param_field(3, "Span (MHz)", f"{state['span_mhz']:.3f}", on_span_text)

    def on_vol_text(text: str) -> None:
        val = parse_float(text)
        if val is None:
            vol_box.set_val(f"{state['audio_gain']:.1f}")
            return
        state["audio_gain"] = float(np.clip(val, 0.1, 20.0))
        vol_box.set_val(f"{state['audio_gain']:.1f}")

    vol_box = add_param_field(4, "Volumen", f"{state['audio_gain']:.1f}", on_vol_text)

    # Fila de acciones
    fig.text(
        CTRL["left"],
        CTRL["row_actions"] + CTRL["btn_h"] + 0.028,
        "Acciones",
        fontsize=CTRL["section_fs"],
        color="#5a6a7a",
        fontweight="bold",
    )

    btn_w = 0.058
    btn_gap = 0.010
    tune_group_left = CTRL["left"]

    def make_step_button(col: int, label: str, delta_mhz: float) -> Button:
        x = tune_group_left + col * (btn_w + btn_gap)
        btn_ax = fig.add_axes([x, CTRL["row_actions"], btn_w, CTRL["btn_h"]])
        btn = Button(btn_ax, label)
        btn.ax.set_facecolor("#e8ecf1")
        btn.label.set_fontsize(9)
        btn.on_clicked(
            lambda _e, d=delta_mhz: set_center_freq_mhz(state["center_freq_hz"] / 1e6 + d)
        )
        return btn

    make_step_button(0, "−1", -1.0)
    make_step_button(1, "+1", 1.0)
    make_step_button(2, "−0.1", -0.1)
    make_step_button(3, "+0.1", 0.1)

    tune_x = tune_group_left + 4 * (btn_w + btn_gap) + 0.012
    tune_ax = fig.add_axes([tune_x, CTRL["row_actions"], 0.072, CTRL["btn_h"]])
    tune_btn = Button(tune_ax, "Auto")
    tune_btn.ax.set_facecolor("#dce8f5")
    tune_btn.label.set_fontsize(9)
    tune_btn.on_clicked(lambda _e: auto_tune_to_peak())

    demod_left = 0.52
    fig.text(
        demod_left + 0.06,
        CTRL["row_actions"] + CTRL["btn_h"] + 0.012,
        "Demod",
        ha="center",
        fontsize=CTRL["label_fs"],
        color="#2c3e50",
    )
    demod_ax = fig.add_axes([demod_left, CTRL["row_actions"], 0.12, CTRL["btn_h"]])
    demod_ax.set_facecolor("#f4f6f8")
    demod_radio = RadioButtons(demod_ax, ("AM", "FM"), active=0)

    def on_demod_change(label: str) -> None:
        state["demod_mode"] = label
        print(f"Demodulación: {label}")

    demod_radio.on_clicked(on_demod_change)

    audio_btn = None
    if audio_player is not None:

        def toggle_audio(_event) -> None:
            if audio_player.enabled:
                audio_player.stop()
                audio_btn.label.set_text("Escuchar")
                print("Audio desactivado.")
            else:
                auto_tune_to_peak()
                audio_player.start()
                audio_player.enabled = True
                for _ in range(3):
                    audio_player.push(make_test_beep())
                audio_btn.label.set_text("Silencio")
                print(
                    f"Audio ON ({state['demod_mode']}). "
                    "¿Oyes un pitido? Si sí, la VM reproduce bien; si no, revisa volumen VM."
                )

            audio_btn.ax.figure.canvas.draw_idle()

        audio_ax = fig.add_axes([0.84, CTRL["row_actions"], 0.12, CTRL["btn_h"]])
        audio_btn = Button(audio_ax, "Escuchar")
        audio_btn.ax.set_facecolor("#d5f0d5")
        audio_btn.label.set_fontsize(9)
        audio_btn.on_clicked(toggle_audio)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.28)
    plt.show(block=False)

    def spectrum_filter_snapshot(samples: np.ndarray) -> np.ndarray:
        """Filtro one-shot para PSD (no comparte estado con el hilo de audio)."""
        sos = design_lowpass_sos(state["bp_bandwidth_hz"], SAMPLE_RATE_HZ)
        if sos is None:
            return samples
        _, baseband, _ = apply_bandpass(
            samples,
            state["bp_center_offset_hz"],
            SAMPLE_RATE_HZ,
            sos,
            [sosfilt_zi(sos), sosfilt_zi(sos)],
        )
        n = np.arange(len(baseband), dtype=np.float64)
        phasor_up = np.exp(
            2j * np.pi * state["bp_center_offset_hz"] * n / SAMPLE_RATE_HZ
        )
        return baseband * phasor_up

    # ===================================
    # BUCLE GUI: solo espectro (audio en hilos aparte)
    # ===================================
    while True:
        now = time.monotonic()
        if now - state["last_psd_t"] >= 1.0 / PSD_UPDATE_HZ:
            state["last_psd_t"] = now
            samples = spectrum_ring.read_latest(PSD_BUFFER_SIZE)
            if samples.size >= FFT_SIZE:
                freqs, psd, psd_db = welch_psd_db(samples)
                state["last_freqs"] = freqs

                if state["psd_db_avg"] is None:
                    state["psd_db_avg"] = psd_db
                else:
                    state["psd_db_avg"] = ALPHA * psd_db + (1 - ALPHA) * state["psd_db_avg"]

                line.set_ydata(state["psd_db_avg"])

                filtered = spectrum_filter_snapshot(samples)
                _, psd_f, psd_f_db = welch_psd_db(filtered)

                if state["psd_filt_db_avg"] is None:
                    state["psd_filt_db_avg"] = psd_f_db
                else:
                    state["psd_filt_db_avg"] = (
                        ALPHA * psd_f_db + (1 - ALPHA) * state["psd_filt_db_avg"]
                    )

                x_mhz = (state["center_freq_hz"] + freqs) / 1e6
                line.set_xdata(x_mhz)
                line_filt.set_xdata(x_mhz)

                in_band = passband_mask_rf(
                    freqs,
                    state["center_freq_hz"],
                    state["bp_center_offset_hz"],
                    state["bp_bandwidth_hz"],
                )
                psd_filt_plot = np.where(in_band, state["psd_filt_db_avg"], np.nan)
                line_filt.set_ydata(psd_filt_plot)

                psd_for_metrics = np.where(in_band, psd_f, 1e-20)
                state["metrics"] = measure_signal(
                    freqs, psd_for_metrics, state["center_freq_hz"]
                )
                metrics_text.set_text(format_metrics(state["metrics"]))

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        if audio_player is not None and audio_player.enabled:
            if now - state["audio_diag_t"] > 2.0:
                state["audio_diag_t"] = now
                pk = getattr(audio_player, "_peak", 0.0)
                qsz = (
                    audio_player._queue.qsize()
                    if hasattr(audio_player, "_queue")
                    else pipeline.audio_queue.qsize()
                )
                print(f"  Audio pico={pk:.3f}  cola={qsz}")

        plt.pause(0.02)
