# ===================================
# CONFIGURACIÓN RTL-SDR
# ===================================
CENTER_FREQ_HZ = 98_000_000   # Frecuencia central inicial: 98 MHz (Radio FM)
SAMPLE_RATE_HZ = 2_400_000   # Tasa de muestreo: 2.4 MS/s
GAIN_DB = 29.7               # Ganancia manual
# Buffers desacoplados: audio vs espectro (PSD >> audio)
N_AUDIO = 32 * 1024                # ~13.6 ms por lectura USB
AUDIO_MERGE_CHUNKS = 4             # ~54 ms por bloque de demodulación
PSD_BUFFER_SIZE = 1024 * 1024      # ~426 ms de IQ para barridos estables
FFT_SIZE = 8192
OVERLAP = FFT_SIZE // 2
ALPHA = 0.35
PSD_UPDATE_HZ = 8.0                # Menos carga CPU → más tiempo para audio

# Rango típico del RTL-SDR (MHz)
FREQ_MIN_MHZ = 24.0
FREQ_MAX_MHZ = 1766.0

# Filtro pasabanda (relativo a la frecuencia central del RTL-SDR)
BP_CENTER_OFFSET_KHZ_DEFAULT = 0.0
BP_BW_KHZ_MIN = 5.0
BP_BW_KHZ_MAX = min(2_000.0, SAMPLE_RATE_HZ / 1e3 * 0.9)
BP_OFFSET_KHZ_MAX = SAMPLE_RATE_HZ / 2 / 1e3

# Audio (demodulación FM → altavoces)
AUDIO_RATE_HZ = 48_000
AUDIO_DECIM = int(SAMPLE_RATE_HZ / AUDIO_RATE_HZ)  # 50 @ 2.4 MS/s
FM_MAX_DEVIATION_HZ = 80_000       # Límite desviación FM (rechaza picos/espurios)
FM_AUDIO_HP_HZ = 300               # Elimina rumble DC/baja frecuencia
FM_AUDIO_LP_HZ = 15_000            # Banda audio FM broadcast
FM_DEEMPH_TAU_S = 75e-6
AUDIO_GAIN_DEFAULT = 2.0
AUDIO_AGC_TARGET = 0.35            # Nivel objetivo AGC lento (evita bombeo)
BP_BANDWIDTH_KHZ_DEFAULT = 200.0   # Canal FM (~±100 kHz audio RF)
FILTER_ORDER = 6                     # Filtros más pronunciados → menos interferencia

# Span = ventana visible del espectro (MHz), centrada en RTL central
SPAN_MHZ_DEFAULT = SAMPLE_RATE_HZ / 1e6
SPAN_MHZ_MIN = 0.05
SPAN_MHZ_MAX = SAMPLE_RATE_HZ / 1e6
