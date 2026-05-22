import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy import signal
import time
import psutil
from matplotlib.widgets import Slider

# --- CONFIGURACIÓN DE HARDWARE ---
def setup_sdr(freq=103.7e6, sample_rate=2.048e6, gain=49.6):
    try:
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.center_freq = freq
        sdr.gain = gain  # Iniciamos en ganancia máxima para interiores
        return sdr
    except Exception as e:
        print(f"Error: No se detectó la RTL-SDR. Verifica la conexión USB.\n{e}")
        return None

def force_redraw(fig):
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def main():
    # --- PARÁMETROS DE ALTA SENSIBILIDAD ---
    fs = 2.048e6
    fc = 103.7e6  # Ajusta a tu emisora local
    # Aumentamos n_samples para filtrar mejor el ruido en interiores
    n_samples = 65536 
    
    sdr = setup_sdr(freq=fc, sample_rate=fs)
    if not sdr: return

    # Configuración de Interfaz
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    fig.canvas.manager.set_window_title('Laboratorio RTL-SDR: MÁXIMA SENSIBILIDAD (Interiores)')
    plt.subplots_adjust(bottom=0.2, hspace=0.4, wspace=0.3)

    # Slider de Ganancia (0 a 49.6 dB)
    ax_gain = plt.axes([0.25, 0.05, 0.5, 0.03])
    s_gain = Slider(ax_gain, 'Ganancia (dB)', 0, 49.6, valinit=49.6, valstep=0.1)

    # Eje de frecuencia relativa
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_samples, 1/fs)) / 1e6

    print("Iniciando modo de alta ganancia... Cierra la ventana para salir.")

    try:
        while plt.fignum_exists(fig.number):
            t_inicio = time.time()
            
            # 1. CAPTURA
            sdr.gain = s_gain.val 
            samples = sdr.read_samples(n_samples)
            
            # --- PROCESAMIENTO CRÍTICO ---
            # Eliminación de DC Offset (Fundamental para ver señales débiles cerca del centro)
            samples = samples - np.mean(samples)
            
            # Detección de Clipping
            clipping = np.any(np.abs(samples.real) >= 0.99) or np.any(np.abs(samples.imag) >= 0.99)

            # 2. CÁLCULOS ESPECTRALES
            # FFT Instantánea
            window = np.hanning(n_samples)
            fft_raw = np.fft.fftshift(np.fft.fft(samples * window))
            mag_db = 20 * np.log10(np.abs(fft_raw) + 1e-10)

            # PSD Periodograma
            f_per, p_per = signal.periodogram(samples, fs, window='hann', scaling='density', return_onesided=False)
            f_per = np.fft.fftshift(f_per) / 1e6
            p_per = np.fft.fftshift(10 * np.log10(p_per + 1e-20))

            # PSD Welch (Configurado para máxima estabilidad en señales débiles)
            # Usamos nperseg más grande para mejor resolución
            f_wel, p_wel = signal.welch(samples, fs, window='hann', nperseg=4096, noverlap=2048, scaling='density', return_onesided=False)
            f_wel = np.fft.fftshift(f_wel) / 1e6
            p_wel = np.fft.fftshift(10 * np.log10(p_wel + 1e-20))
            
            noise_floor = np.median(p_wel)

            # 3. ACTUALIZACIÓN DE GRÁFICAS
            # FFT
            axs[0,0].clear()
            axs[0,0].plot(freq_axis, mag_db, color='#1f77b4', lw=0.5)
            axs[0,0].set_title(f"FFT (Clipping: {clipping})", color='red' if clipping else 'black')
            axs[0,0].set_ylim([-20, 70]) # Ajustado para ver picos altos
            axs[0,0].grid(True, alpha=0.3)

            # Periodograma
            axs[0,1].clear()
            axs[0,1].plot(f_per, p_per, color='#2ca02c', lw=0.7)
            axs[0,1].set_title("PSD Periodograma")
            axs[0,1].grid(True, alpha=0.3)

            # Welch (Recomendado para interiores)
            axs[1,0].clear()
            axs[1,0].plot(f_wel, p_wel, color='#d62728', lw=1.2)
            axs[1,0].axhline(noise_floor, color='black', linestyle='--', alpha=0.6, label=f'Ruido: {noise_floor:.1f} dB')
            axs[1,0].set_title("PSD Welch (Señal Filtrada)")
            axs[1,0].set_xlabel("Frecuencia (MHz)")
            axs[1,0].legend(loc='upper right')
            axs[1,0].grid(True, alpha=0.3)

            # Costo y Estado
            cpu_usage = psutil.Process().cpu_percent()
            axs[1,1].clear()
            axs[1,1].axis('off')
            info_text = (
                f"--- ESTADO DEL SISTEMA ---\n\n"
                f"Ganancia: {sdr.gain:.1f} dB\n"
                f"Muestras: {n_samples}\n"
                f"CPU: {cpu_usage}%\n\n"
                f"MODO: MÁXIMA SENSIBILIDAD\n"
                f"Estado: {'SATURADO' if clipping else 'Capturando...'}"
            )
            axs[1,1].text(0.1, 0.5, info_text, fontsize=11, family='monospace', 
                          bbox=dict(facecolor='yellow' if clipping else 'white', alpha=0.3))

            force_redraw(fig)
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nPrograma finalizado.")
    finally:
        sdr.close()

if __name__ == "__main__":
    main()