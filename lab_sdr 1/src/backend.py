import queue
import shutil
import subprocess
import threading
import time

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfilt, sosfilt_zi, welch
from rtlsdr import RtlSdr
from rtlsdr.rtlsdr import LibUSBError

from src.config import (
    AUDIO_AGC_TARGET,
    AUDIO_DECIM,
    AUDIO_GAIN_DEFAULT,
    AUDIO_MERGE_CHUNKS,
    AUDIO_RATE_HZ,
    BP_BANDWIDTH_KHZ_DEFAULT,
    BP_CENTER_OFFSET_KHZ_DEFAULT,
    BP_BW_KHZ_MAX,
    BP_BW_KHZ_MIN,
    BP_OFFSET_KHZ_MAX,
    CENTER_FREQ_HZ,
    FILTER_ORDER,
    FM_AUDIO_HP_HZ,
    FM_AUDIO_LP_HZ,
    FM_DEEMPH_TAU_S,
    FM_MAX_DEVIATION_HZ,
    FFT_SIZE,
    GAIN_DB,
    N_AUDIO,
    OVERLAP,
    PSD_BUFFER_SIZE,
    PSD_UPDATE_HZ,
    SAMPLE_RATE_HZ,
    SPAN_MHZ_DEFAULT,
)

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

HAS_PAPLAY = shutil.which("paplay") is not None
HAS_AUDIO = HAS_SOUNDDEVICE or HAS_PAPLAY


def parse_float(text: str) -> float | None:
    try:
        return float(text.strip().replace(",", "."))
    except ValueError:
        return None


def _usb_troubleshooting() -> None:
    print(
        "\n--- Diagnóstico USB (LIBUSB_ERROR_IO) ---\n"
        "El dongle se ve en USB pero no responde al abrirlo. Suele ser:\n"
        "  • VirtualBox: Extension Pack instalado, filtro USB para 0bda:2838, "
        "conectar el dongle a la VM (no solo al host).\n"
        "  • Desconectar y volver a conectar el RTL-SDR; probar otro puerto USB 2.0.\n"
        "  • Cerrar otros programas que usen el SDR (GQRX, CubicSDR, rtl_test).\n"
        "  • En Linux, evitar el driver DVB: sudo modprobe -r dvb_usb_rtl28xxu\n"
        "  • Probar en el host (fuera de la VM): python main.py\n"
        "Comprueba: lsusb | grep -i 2838   y   rtl_test -t\n"
    )


def open_rtlsdr(device_index: int = 0) -> RtlSdr:
    """Abre el RTL-SDR con mensajes claros si falla la comunicación USB."""
    try:
        serials = RtlSdr.get_device_serial_addresses()
    except LibUSBError:
        serials = []

    if not serials:
        raise SystemExit(
            "No se detectó ningún RTL-SDR (0 dispositivos).\n"
            "Conecta el dongle y, si usas VirtualBox, asígnalo a la VM con un filtro USB."
        )

    print(f"Dongles detectados ({len(serials)}): {', '.join(serials) or '(sin número de serie)'}")

    try:
        return RtlSdr(device_index=device_index)
    except LibUSBError as err:
        print(f"Error al abrir RTL-SDR (índice {device_index}): {err}")
        _usb_troubleshooting()
        raise SystemExit(1) from err


def design_lowpass_sos(bandwidth_hz: float, fs_hz: float, order: int = FILTER_ORDER):
    """Paso bajo para usar tras mezclar la señal al centro del pasabanda."""
    cutoff = min(bandwidth_hz / 2, fs_hz / 2 - 1.0)
    if cutoff <= 0:
        return None
    return butter(order, cutoff, btype="low", fs=fs_hz, output="sos")


def apply_bandpass(
    samples: np.ndarray,
    center_offset_hz: float,
    fs_hz: float,
    sos_lp,
    zi,
) -> tuple[np.ndarray, np.ndarray, list | None]:
    """
    Pasabanda IQ: mezcla a banda base → paso bajo → remezcla.
    Devuelve (señal en RF para espectro, banda base para demodulación, estado zi).
    """
    if sos_lp is None or zi is None:
        return samples, samples, zi

    n = np.arange(len(samples), dtype=np.float64)
    phasor_dn = np.exp(-2j * np.pi * center_offset_hz * n / fs_hz)
    mixed = samples * phasor_dn

    i_out, zi_i = sosfilt(sos_lp, mixed.real, zi=zi[0])
    q_out, zi_q = sosfilt(sos_lp, mixed.imag, zi=zi[1])
    baseband = i_out + 1j * q_out

    phasor_up = np.exp(2j * np.pi * center_offset_hz * n / fs_hz)
    return baseband * phasor_up, baseband, [zi_i, zi_q]


def fm_demod(iq: np.ndarray, fs_hz: float) -> np.ndarray:
    """Demodulación FM con limitación de desviación (rechaza impulsos de interferencia)."""
    dphi = np.angle(iq[1:] * np.conj(iq[:-1]))
    freq_hz = dphi * fs_hz / (2.0 * np.pi)
    np.clip(freq_hz, -FM_MAX_DEVIATION_HZ, FM_MAX_DEVIATION_HZ, out=freq_hz)
    return freq_hz.astype(np.float32)


_AUDIO_HP_FM = butter(2, FM_AUDIO_HP_HZ, btype="high", fs=AUDIO_RATE_HZ, output="sos")
_AUDIO_LP_FM = butter(4, FM_AUDIO_LP_HZ, btype="low", fs=AUDIO_RATE_HZ, output="sos")
_AUDIO_LP_AM = butter(4, 6_000, btype="low", fs=AUDIO_RATE_HZ, output="sos")
_AUDIO_DEEMPH_B = [1.0 - np.exp(-1.0 / (AUDIO_RATE_HZ * FM_DEEMPH_TAU_S))]
_AUDIO_DEEMPH_A = [1.0, -np.exp(-1.0 / (AUDIO_RATE_HZ * FM_DEEMPH_TAU_S))]


def am_demod(iq: np.ndarray) -> np.ndarray:
    """Demodulación AM (detector de envolvente)."""
    return np.abs(iq).astype(np.float32)


class AudioProcessor:
    """Cadena de audio con estado (AGC lento, filtros continuos, menos chasquidos)."""

    def __init__(self) -> None:
        self.agc_level = 1e-3
        self._deemph_zi = np.zeros(max(len(_AUDIO_DEEMPH_A), len(_AUDIO_DEEMPH_B)) - 1)
        self._hp_zi = sosfilt_zi(_AUDIO_HP_FM)
        self._lp_fm_zi = sosfilt_zi(_AUDIO_LP_FM)
        self._lp_am_zi = sosfilt_zi(_AUDIO_LP_AM)

    def _agc(self, audio: np.ndarray, gain: float) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) + 1e-9
        if peak > self.agc_level:
            self.agc_level = 0.08 * peak + 0.92 * self.agc_level
        else:
            self.agc_level = 0.002 * peak + 0.998 * self.agc_level
        scale = gain * AUDIO_AGC_TARGET / max(self.agc_level, 1e-6)
        return np.clip(audio * scale, -1.0, 1.0).astype(np.float32)

    def process(self, baseband: np.ndarray, gain: float, mode: str) -> np.ndarray:
        if baseband.size < 8:
            return np.zeros(0, dtype=np.float32)

        baseband = baseband - np.mean(baseband)

        if mode == "FM":
            audio = fm_demod(baseband, SAMPLE_RATE_HZ)
            audio = resample_poly(audio, 1, AUDIO_DECIM).astype(np.float32)
            audio, self._hp_zi = sosfilt(_AUDIO_HP_FM, audio, zi=self._hp_zi)
            audio, self._lp_fm_zi = sosfilt(_AUDIO_LP_FM, audio, zi=self._lp_fm_zi)
            audio, self._deemph_zi = lfilter(
                _AUDIO_DEEMPH_B, _AUDIO_DEEMPH_A, audio, zi=self._deemph_zi
            )
            audio = audio.astype(np.float32)
        else:
            audio = am_demod(baseband)
            audio = resample_poly(audio, 1, AUDIO_DECIM).astype(np.float32)
            audio, self._lp_am_zi = sosfilt(_AUDIO_LP_AM, audio, zi=self._lp_am_zi)

        return self._agc(audio, gain)


def iq_to_audio(
    baseband: np.ndarray,
    gain: float,
    mode: str,
    processor: AudioProcessor | None = None,
) -> np.ndarray:
    proc = processor if processor is not None else AudioProcessor()
    return proc.process(baseband, gain, mode)


def make_test_beep(duration_s: float = 0.4, freq_hz: float = 440.0) -> np.ndarray:
    """Tono de prueba para verificar que la salida de audio de la VM funciona."""
    n = int(AUDIO_RATE_HZ * duration_s)
    t = np.arange(n, dtype=np.float32) / AUDIO_RATE_HZ
    return (0.35 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


class PaplayAudioPlayer:
    """Salida por PulseAudio/PipeWire (paplay) si no hay sounddevice."""

    def __init__(self, sample_rate: int = AUDIO_RATE_HZ) -> None:
        self.sample_rate = sample_rate
        self.enabled = False
        self._proc: subprocess.Popen | None = None
        self._peak = 0.0

    def start(self) -> None:
        if not HAS_PAPLAY or self._proc is not None:
            return
        cmd = [
            "paplay",
            f"--rate={self.sample_rate}",
            "--format=float32le",
            "--channels=1",
            "--raw",
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print("Audio vía paplay (PulseAudio).")

    def stop(self) -> None:
        self.enabled = False
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=1)
            except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
                self._proc.kill()
            self._proc = None

    def push(self, audio: np.ndarray) -> None:
        if not self.enabled or self._proc is None or audio.size == 0:
            return
        self._peak = max(self._peak * 0.95, float(np.max(np.abs(audio))))
        try:
            self._proc.stdin.write(audio.tobytes())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.stop()


class SounddeviceAudioPlayer:
    """Reproductor con sounddevice y buffer continuo."""

    def __init__(self, sample_rate: int = AUDIO_RATE_HZ) -> None:
        self.sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self.enabled = False
        self._stream = None
        self._peak = 0.0

    def _callback(self, outdata, frames, _time, status) -> None:
        if status:
            print(f"Audio: {status}")
        with self._lock:
            while self._buffer.size < frames:
                try:
                    chunk = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._buffer = np.concatenate((self._buffer, chunk))

            if self._buffer.size >= frames:
                outdata[:, 0] = self._buffer[:frames]
                self._buffer = self._buffer[frames:]
            else:
                outdata.fill(0)

    def start(self) -> None:
        if not HAS_SOUNDDEVICE or self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=512,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        print(f"Audio vía sounddevice → {sd.query_devices(sd.default.device[1], 'output')['name']}")

    def stop(self) -> None:
        self.enabled = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def push(self, audio: np.ndarray) -> None:
        if not self.enabled or self._stream is None or audio.size == 0:
            return
        self._peak = max(self._peak * 0.95, float(np.max(np.abs(audio))))
        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(audio)


def create_audio_player() -> PaplayAudioPlayer | SounddeviceAudioPlayer | None:
    if HAS_SOUNDDEVICE:
        return SounddeviceAudioPlayer()
    if HAS_PAPLAY:
        return PaplayAudioPlayer()
    return None


def passband_mask_rf(
    freqs_baseband_hz: np.ndarray,
    rf_center_hz: float,
    bp_center_offset_hz: float,
    bp_bandwidth_hz: float,
) -> np.ndarray:
    """Máscara booleana en el eje RF absoluto del espectro."""
    rf_hz = rf_center_hz + freqs_baseband_hz
    fc = rf_center_hz + bp_center_offset_hz
    half = bp_bandwidth_hz / 2
    return (rf_hz >= fc - half) & (rf_hz <= fc + half)


def welch_psd_db(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PSD en banda base: frecuencias relativas (Hz), PSD lineal y PSD en dB."""
    freqs, psd = welch(
        samples,
        fs=SAMPLE_RATE_HZ,
        window="hann",
        nperseg=FFT_SIZE,
        noverlap=OVERLAP,
        return_onesided=False,
        scaling="density",
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
    return freqs, psd, psd_db


def measure_signal(freqs_hz: np.ndarray, psd: np.ndarray, rf_center_hz: float) -> dict:
    """
    Métricas espectrales de la señal filtrada.
    freqs_hz: eje en banda base; rf_center_hz: centro del RTL-SDR.
    """
    psd = np.maximum(psd, 1e-20)
    psd_db = 10 * np.log10(psd)
    rf_hz = rf_center_hz + freqs_hz

    peak_idx = int(np.argmax(psd_db))
    peak_db = psd_db[peak_idx]
    threshold_db = peak_db - 3.0

    above = psd_db >= threshold_db
    if not np.any(above):
        return {
            "fc_hz": rf_center_hz,
            "bw_hz": 0.0,
            "power_w": float(np.sum(psd) * (freqs_hz[1] - freqs_hz[0])),
            "power_db": -np.inf,
        }

    idx = np.where(above)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, splits)
    seg = next((s for s in segments if s[0] <= peak_idx <= s[-1]), segments[0])
    lo, hi = int(seg[0]), int(seg[-1])

    bw_hz = float(rf_hz[hi] - rf_hz[lo])
    fc_hz = float(np.sum(rf_hz * psd) / np.sum(psd))
    df = float(freqs_hz[1] - freqs_hz[0]) if len(freqs_hz) > 1 else 1.0
    power_w = float(np.sum(psd) * df)

    return {
        "fc_hz": fc_hz,
        "bw_hz": bw_hz,
        "power_w": power_w,
        "power_db": 10 * np.log10(power_w) if power_w > 0 else -np.inf,
    }


def freq_axis_mhz(center_hz: float) -> np.ndarray:
    """Eje de frecuencias RF absolutas en MHz para el ancho de banda actual."""
    lo = (center_hz - SAMPLE_RATE_HZ / 2) / 1e6
    hi = (center_hz + SAMPLE_RATE_HZ / 2) / 1e6
    return np.linspace(lo, hi, FFT_SIZE, endpoint=False)


class IQRingBuffer:
    """Buffer circular IQ para el espectro (mucho mayor que el bloque de audio)."""

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.complex64)
        self._lock = threading.Lock()
        self._write = 0
        self._filled = 0
        self.capacity = capacity

    def write(self, chunk: np.ndarray) -> None:
        n = int(len(chunk))
        if n <= 0:
            return
        if n >= self.capacity:
            chunk = chunk[-self.capacity :]
            n = self.capacity

        with self._lock:
            end = self._write + n
            if end <= self.capacity:
                self._buf[self._write : end] = chunk
            else:
                first = self.capacity - self._write
                self._buf[self._write :] = chunk[:first]
                self._buf[: end - self.capacity] = chunk[first:]
            self._write = end % self.capacity
            self._filled = min(self._filled + n, self.capacity)

    def read_latest(self, n: int) -> np.ndarray:
        n = min(int(n), self._filled, self.capacity)
        if n <= 0:
            return np.array([], dtype=np.complex64)

        with self._lock:
            start = (self._write - n) % self.capacity
            if start + n <= self.capacity:
                return self._buf[start : start + n].copy()
            first = self.capacity - start
            return np.concatenate((self._buf[start:], self._buf[: n - first]))


class SDRPipeline:
    """
    Captura continua en hilo dedicado.
    Audio: bloques pequeños en cola. Espectro: anillo grande independiente.
    """

    def __init__(self, sdr: RtlSdr, spectrum_ring: IQRingBuffer) -> None:
        self.sdr = sdr
        self.spectrum_ring = spectrum_ring
        self.audio_player = None
        self._audio_enabled = lambda: False
        self._audio_gain = lambda: AUDIO_GAIN_DEFAULT
        self._audio_demod = lambda: "AM"
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.params_lock = threading.Lock()
        self.bp_center_offset_hz = BP_CENTER_OFFSET_KHZ_DEFAULT * 1e3
        self.bp_bandwidth_hz = BP_BANDWIDTH_KHZ_DEFAULT * 1e3
        self.bp_lp_sos = design_lowpass_sos(self.bp_bandwidth_hz, SAMPLE_RATE_HZ)
        self._audio_zi = None
        self._audio_accum: list[np.ndarray] = []
        self._audio_processor = AudioProcessor()
        self._reset_audio_filter()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)

    def bind_audio(
        self,
        player,
        enabled_fn,
        gain_fn,
        demod_fn,
    ) -> None:
        self.audio_player = player
        self._audio_enabled = enabled_fn
        self._audio_gain = gain_fn
        self._audio_demod = demod_fn

    def _reset_audio_filter(self) -> None:
        if self.bp_lp_sos is None:
            self._audio_zi = None
        else:
            zi0 = sosfilt_zi(self.bp_lp_sos)
            self._audio_zi = [zi0.copy(), zi0.copy()]

    def update_filter(self, center_offset_hz: float, bandwidth_hz: float, *, reset: bool) -> None:
        with self.params_lock:
            self.bp_center_offset_hz = center_offset_hz
            self.bp_bandwidth_hz = bandwidth_hz
            self.bp_lp_sos = design_lowpass_sos(bandwidth_hz, SAMPLE_RATE_HZ)
            if reset:
                self._reset_audio_filter()

    def start(self) -> None:
        self._capture_thread.start()
        self._audio_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._capture_thread.join(timeout=2.0)
        self._audio_thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.sdr.read_samples(N_AUDIO)
            except Exception as err:
                if not self.stop_event.is_set():
                    print(f"Error lectura RTL-SDR: {err}")
                break
            chunk = chunk - np.mean(chunk)
            self.spectrum_ring.write(chunk)
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass

    def _audio_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self.audio_player is None or not self._audio_enabled():
                self._audio_accum.clear()
                continue

            self._audio_accum.append(chunk)
            if len(self._audio_accum) < AUDIO_MERGE_CHUNKS:
                continue

            merged = np.concatenate(self._audio_accum)
            self._audio_accum.clear()

            with self.params_lock:
                offset = self.bp_center_offset_hz
                sos = self.bp_lp_sos
                zi = self._audio_zi

            _, baseband, new_zi = apply_bandpass(
                merged, offset, SAMPLE_RATE_HZ, sos, zi
            )
            with self.params_lock:
                self._audio_zi = new_zi

            audio = self._audio_processor.process(
                baseband, self._audio_gain(), self._audio_demod()
            )
            if audio.size > 0:
                self.audio_player.push(audio)


def initialize_sdr_system() -> tuple[
    RtlSdr,
    IQRingBuffer,
    SDRPipeline,
    PaplayAudioPlayer | SounddeviceAudioPlayer | None,
    dict,
]:
    """
    Inicializa RTL-SDR, pipeline de captura/audio y estado compartido con la GUI.
    """
    sdr = open_rtlsdr()
    sdr.sample_rate = SAMPLE_RATE_HZ
    sdr.center_freq = CENTER_FREQ_HZ
    sdr.gain = GAIN_DB

    spectrum_ring = IQRingBuffer(PSD_BUFFER_SIZE)
    pipeline = SDRPipeline(sdr, spectrum_ring)

    state = {
        "center_freq_hz": CENTER_FREQ_HZ,
        "psd_db_avg": None,
        "psd_filt_db_avg": None,
        "bp_center_offset_hz": BP_CENTER_OFFSET_KHZ_DEFAULT * 1e3,
        "bp_bandwidth_hz": BP_BANDWIDTH_KHZ_DEFAULT * 1e3,
        "span_mhz": SPAN_MHZ_DEFAULT,
        "metrics": None,
    }

    print("RTL-SDR configurada correctamente.")
    print(f"Frecuencia Central: {state['center_freq_hz'] / 1e6:.3f} MHz")
    print(f"Tasa de Muestreo: {SAMPLE_RATE_HZ / 1e6:.3f} MS/s")
    print(f"Ganancia: {GAIN_DB:.1f} dB")
    audio_ms = N_AUDIO * AUDIO_MERGE_CHUNKS / SAMPLE_RATE_HZ * 1e3
    print(
        f"Audio: {N_AUDIO / SAMPLE_RATE_HZ * 1e3:.1f} ms × {AUDIO_MERGE_CHUNKS} "
        f"= {audio_ms:.0f} ms/bloque | "
        f"PSD: {PSD_BUFFER_SIZE / SAMPLE_RATE_HZ * 1e3:.0f} ms @ {PSD_UPDATE_HZ:.0f} Hz"
    )
    if HAS_SOUNDDEVICE:
        print("Audio FM: pulsa «Escuchar» (sounddevice).")
    elif HAS_PAPLAY:
        print("Audio FM: pulsa «Escuchar» (paplay / PulseAudio).")
    else:
        print("Sin audio. Instala: pip install sounddevice   o   apt install pulseaudio-utils")
    print("Panel inferior: entradas de texto (Enter para aplicar). Ctrl+C para detener.")

    audio_player = create_audio_player()
    state["audio_gain"] = AUDIO_GAIN_DEFAULT
    state["demod_mode"] = "AM"
    state["audio_diag_t"] = 0.0
    state["last_freqs"] = None
    state["last_psd_t"] = 0.0

    if audio_player is not None:
        pipeline.bind_audio(
            audio_player,
            enabled_fn=lambda: audio_player.enabled,
            gain_fn=lambda: state["audio_gain"],
            demod_fn=lambda: state["demod_mode"],
        )
    pipeline.start()

    return sdr, spectrum_ring, pipeline, audio_player, state


def shutdown_sdr_system(
    pipeline: SDRPipeline | None,
    audio_player: PaplayAudioPlayer | SounddeviceAudioPlayer | None,
    sdr: RtlSdr | None,
) -> None:
    if pipeline is not None:
        pipeline.stop()
    if audio_player is not None:
        audio_player.stop()
    if sdr is not None:
        sdr.close()
        print("RTL-SDR liberada.")
