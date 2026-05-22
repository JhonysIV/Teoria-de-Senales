#!/usr/bin/env python3
"""Punto de entrada: inicializa backend (SDR/audio) y arranca la GUI."""

from src.backend import initialize_sdr_system, shutdown_sdr_system
from src.frontend import run_gui


def main() -> None:
    sdr = None
    audio_player = None
    pipeline = None
    try:
        sdr, spectrum_ring, pipeline, audio_player, state = initialize_sdr_system()
        run_gui(sdr, spectrum_ring, pipeline, audio_player, state)
    except KeyboardInterrupt:
        print("\nDeteniendo la adquisición...")
    finally:
        shutdown_sdr_system(pipeline, audio_player, sdr)


if __name__ == "__main__":
    main()
