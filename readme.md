# Implementación de un Receptor Digital de Radio FM mediante RTL-SDR y Python

##  Introducción y Contexto

Este proyecto fue desarrollado en el marco del Laboratorio de SDR (Software Defined Radio). El objetivo principal fue la construcción de un receptor de radio funcional capaz de capturar señales del espectro electromagnético en tiempo real, procesarlas digitalmente mediante algoritmos de DSP (Digital Signal Processing) y obtener una salida de audio inteligible junto con una representación visual de la densidad espectral de potencia (PSD).

Aunque el código base permite demodulación AM, el ejercicio se centró exclusivamente en la Banda de FM Comercial, permitiendo analizar emisoras locales y entender el comportamiento de las señales de banda ancha.

---

#  Evolución del Entorno de Trabajo: De WSL a Linux Mint

Una parte fundamental del aprendizaje en este laboratorio fue la configuración del entorno:

## 1. Fase Inicial (WSL)

Se intentó inicialmente trabajar bajo WSL (Windows Subsystem for Linux). Sin embargo, debido a las capas de abstracción de red y USB (`usbipd-win`), se experimentaron latencias en el audio y dificultades en el reconocimiento consistente del hardware RTL-SDR.

## 2. Fase Final (Linux Mint)

Para garantizar un acceso directo al hardware (Raw USB) y minimizar la latencia de procesamiento de la tarjeta de sonido, se migró a una distribución nativa de Linux Mint. En este entorno, el uso de los drivers `rtl-sdr` y la comunicación con el servidor de audio (`PulseAudio/PipeWire`) fue significativamente más estable, permitiendo una ejecución fluida del código y una escucha sin cortes.

---

# Arquitectura del Sistema

El sistema se divide en tres bloques lógicos implementados en el script `lab_sdr.py`.

---

## 1. Captura de Datos (Front-end de Radio)

Utilizamos un dongle RTL-SDR que actúa como un digitalizador de alta velocidad.

### Características principales

- **Muestreo IQ:**  
  Capturamos muestras complejas a 2.4 MS/s. Estas muestras contienen tanto la magnitud como la fase de la señal de radio.

- **Multihilo (Threading):**  
  El script utiliza hilos independientes. Mientras un hilo captura datos del USB y los deposita en un Ring Buffer (buffer circular), otros hilos se encargan de la visualización y el audio. Esto evita que la interfaz gráfica "congele" el sonido.

---

## 2. Procesamiento Digital de Señales (DSP) para FM

El procesamiento de FM sigue estos pasos críticos:

### • Mezclado Digital

La señal de interés se desplaza hacia la frecuencia de banda base (0 Hz).

### • Filtro de Selección de Canal

Se aplica un filtro Butterworth de orden 6 (vía `scipy.signal.sosfilt`) con un ancho de banda de 190-250 kHz, ideal para capturar la señal completa de una emisora FM (incluyendo subportadoras).

### • Demodulación por Discriminación de Fase

La FM transporta información en los cambios de frecuencia. El código calcula la diferencia de fase entre muestras consecutivas utilizando:

```python
np.angle(iq[1:] * np.conj(iq[:-1]))
```

### • De-énfasis

Se aplica un filtro para revertir la pre-acentuación de agudos que las emisoras FM aplican en la transmisión, restaurando el balance tonal original.

---

## 3. Visualización Espectral

Para el análisis visual, se implementa el Método de Welch. Este método promedia varios periodogramas de la señal capturada, lo que resulta en una gráfica de la PSD (Power Spectral Density) mucho más suave y fácil de interpretar que una simple FFT, permitiendo identificar claramente dónde están las portadoras de las emisoras.

---

# Análisis de Resultados en Laboratorio

En las pruebas realizadas (como se observa en la captura de pantalla adjunta), se sintonizó la frecuencia **105.7 MHz**.

### Resultados obtenidos

- **Frecuencia Central:**  
  105.700 MHz.

- **Potencia Medida:**  
  Aproximadamente -22.7 dB, lo que indica una recepción fuerte y clara.

- **Ancho de Banda de Vista (Span):**  
  2.4 MHz, permitiendo ver no solo la emisora sintonizada sino también el ruido de fondo y posibles señales adyacentes.

- **Resultado General:**  
  Se logró una escucha estable de la señal de FM, con una visualización en tiempo real que permitía ajustar el filtro dinámicamente para eliminar interferencias de canales cercanos.

---

# Instalación en Linux Mint

Para replicar este experimento en una distribución base Debian/Ubuntu como Linux Mint, siga estos pasos.

---

## 1. Instalar dependencias del sistema

```bash
sudo apt update
sudo apt install rtl-sdr librtlsdr-dev python3-tk python3-pip
```

---

## 2. Configurar privilegios de USB (Udev)

Para usar el SDR sin ser root, crear el archivo:

```bash
/etc/udev/rules.d/20-rtlsdr.rules
```

Con el siguiente contenido:

```bash
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"
```

---

## 3. Instalar librerías de Python

```bash
pip install pyrtlsdr==0.3.0 numpy scipy matplotlib sounddevice
```

---

## 4. Ejecución

```bash
python lab_sdr.py
```

---

# Instrucciones de Uso de la Interfaz

- **RTL Central:**  
  Define la frecuencia que el aparato "escucha" físicamente.

- **Δf filtro:**  
  Permite mover el filtro verde para sintonizar una emisora que no esté exactamente en el centro.

- **Ancho BW:**  
  Para FM comercial, se recomienda entre 180 y 250 kHz.

- **Botón "Auto":**  
  Escanea el espectro visible y centra el filtro en el pico más potente (la emisora más fuerte).

- **Escuchar:**  
  Activa la salida de audio. Si se escucha un pitido inicial, significa que el sistema de audio está correctamente sincronizado.

---

# Información del Proyecto

- **Desarrollado por:** Laboratorio de SDR  
- **Fecha:** Mayo 2026  
- **Hardware:** RTL2832U V3  
- **Software:** Python 3.x / Linux Mint 21+
