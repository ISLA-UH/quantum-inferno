import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from quantum_inferno.synth import benchmark_signals
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon
import quantum_inferno.plot_templates.plot_templates as pt
import quantum_inferno.plot_templates.plot_base as pb


if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows. Added taper, noise, and AA to signal.
    Added STFT Tukey window alpha > 0.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """
    EVENT_NAME = "tone test"
    frequency_tone_hz = 60
    [
        mic_sig,
        time_s,
        time_fft_nd,
        frequency_sample_rate_hz,
        frequency_center_fft_hz,
        frequency_resolution_fft_hz,
    ] = benchmark_signals.well_tempered_tone(
        frequency_center_hz=frequency_tone_hz,
        frequency_sample_rate_hz=800,
        time_duration_s=16,
        time_fft_s=1,
        use_fft_frequency=True,
        add_noise_taper_aa=True,
        output_desc=True
    )
    alpha = 0.25  # 25% Tukey (Cosine) window

    # Compute the spectrogram with the spectrum option
    frequency_spect_hz, time_spect_s, psd_spec_power = signal.spectrogram(
        x=mic_sig,
        fs=frequency_sample_rate_hz,
        window=("tukey", alpha),
        nperseg=time_fft_nd,
        noverlap=time_fft_nd // 2,
        nfft=time_fft_nd,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        scaling="spectrum",
        mode="psd",
    )

    # Express in bits; revisit
    mic_spect_bits = to_log2_with_epsilon(np.sqrt(psd_spec_power))

    wf_base = pb.WaveformPlotBase(station_id="00", figure_title=f"waveform for {EVENT_NAME}", start_time_epoch=0)
    wf_panel_a = pb.WaveformPanel(mic_sig, time_s, units="Norm")
    fig = pt.plot_wf_3_vert(wf_base, wf_panel_a, wf_panel_a, wf_panel_a)

    fmin = 2 * frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz / 2  # Nyquist
    mesh_base = pb.MeshBase(time_spect_s, frequency_spect_hz, frequency_hz_ymin=fmin, frequency_hz_ymax=fmax)
    mesh_panel = pb.MeshPanel(mic_spect_bits, "range", cbar_units="bits")

    fig2 = pt.plot_n_mesh_wf_vert(mesh_base, [mesh_panel, mesh_panel], wf_base, wf_panel_a)

    plt.show()
