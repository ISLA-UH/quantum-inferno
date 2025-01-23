"""
Quantum inferno example: s00_vertical_wf_mesh_plots.py
Tutorial on the quantum-inferno vertical waveform and spectrogram plotting functions and their parameters.
"""
import numpy as np
import matplotlib.pyplot as plt

import quantum_inferno.plot_templates.plot_base as ptb
import quantum_inferno.utilities.short_time_fft as stft
from quantum_inferno.plot_templates.plot_templates import plot_n_mesh_wf_vert
from quantum_inferno.plot_templates.plot_templates_examples import plot_wf_mesh_mesh_vert_example, plot_wf_mesh_vert_example
from quantum_inferno.plot_templates.figure_attributes import AudioParams, AspectRatioType
from quantum_inferno.synth import benchmark_signals
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

print(__doc__)


def calculate_t_domain_example_values():
    # The values calculated here to be plotted are adapted from the examples, refer to s02_tone_stft_vs_spectrogram.py
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
    )
    return frequency_resolution_fft_hz, frequency_sample_rate_hz, mic_sig, time_s, time_fft_nd


def calculate_tf_domain_example_values(mic_sig, time_fft_nd, frequency_sample_rate_hz, frequency_resolution_fft_hz,
                                       alpha=0.25):
    frequency_stft_hz, time_stft_s, stft_complex = stft.get_stft_tukey_mag(
        timeseries=mic_sig,
        sample_rate_hz=frequency_sample_rate_hz,
        tukey_alpha=alpha,
        segment_length=time_fft_nd,
        overlap_length=time_fft_nd // 2,  # 50% overlap
        scaling="magnitude",
        padding="zeros",
    )
    stft_power = 2 * np.abs(stft_complex) ** 2
    mic_stft_bits = to_log2_with_epsilon(stft_power)
    fmin = 2 * frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz / 2  # Nyquist
    return time_stft_s, frequency_stft_hz, mic_stft_bits, fmin, fmax


def main():
    print("\nWe'll start by generating the times series and STFT arrays from s02_tone_stft_vs_spectrogram.py.")
    # Calculate some values to be plotted (see s02_tone_stft_vs_spectrogram.py for details on the values themselves)
    [freq_resolution_fft_hz, freq_sample_rate_hz, mic_sig, time_s, time_fft_nd] = calculate_t_domain_example_values()
    [time_stft_s, frequency_stft_hz, mic_stft_bits, fmin, fmax] = \
        calculate_tf_domain_example_values(mic_sig, time_fft_nd, freq_sample_rate_hz, freq_resolution_fft_hz)
    # # # EXAMPLE: DEFAULT PARAMETERS # # #
    print("\nAll the plot templates are built on Base and Panel objects defined in the plot_base.py file. We'll start")
    print("by building Base and Panel objects with only the required variables, then explore parameters.")
    print("First, let's give WaveformPlotBase a figure title and a blank station id since our data is synthetic.")
    # create WaveformPlotBase object with only required parameters
    wf_base = ptb.WaveformPlotBase(station_id="", figure_title="2-panel plot with default parameters")
    print("Next, we'll build the waveform panel with the synthetic microphone signal and time values.")
    # create WaveformPanel object with only required parameters
    wf_panel = ptb.WaveformPanel(mic_sig, time_s)
    print("\nThe MeshBase and MeshPanel objects parallel the Waveform objects, but the parameters are different.")
    print("For the MeshBase object, we'll use the time and frequency values from the example STFT calculation.")
    # create MeshBase object with only required parameters
    mesh_base = ptb.MeshBase(time=time_stft_s, frequency=frequency_stft_hz)
    print("For the MeshPanel object, we only need to input the STFT power values.")
    # create MeshPanel object with only required parameters
    mesh_panel = ptb.MeshPanel(tfr=mic_stft_bits)
    print("\nThe workhorse of the plotting functions is plot_n_mesh_wf_vert, which creates a figure with vertical")
    print("subplots for each MeshPanel it's given and a single WaveformPanel at the bottom. We'll use this function to")
    print("plot the example data. Close the figure when you're ready to continue.")
    # plot the example data
    print("\n\t\t\t PLOTTING EXAMPLE FIGURE: DEFAULT PARAMETERS")
    stft = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    plt.show()
    # # # CUSTOMIZED BASE AND PANEL PARAMETERS # # #
    print("\nFor a more customized figure, we can specify additional parameters in the Base and Panel objects.")
    print("In wf_base, we can change the following parameters:")
    print("\t- figure_title_show: bool, if True, show the figure title.  Default True")
    print("\t- start_time_epoch: float, the epoch start time of the data.  Default 0.")
    print("\t- params_tfr: AudioParams, parameters for plotting audio data.  Default AudioParams()")
    print("\t- units_time: str, label of units for time component.  Default 's'")
    print("\t- label_panel_show: bool, if True, show the label.  Default False")
    print("\t- labels_fontweight: optional str, font weight of the labels.  Default 'bold'")
    print("\t- waveform_color: optional str, color of the waveform.  Default None")
    print("We'll turn on the panel labels, change the waveform color, and update the title.")
    # set wf_base optional parameters
    wf_base.label_panel_show = True
    wf_base.waveform_color = "#1f77b4"  # Default color for matplotlib
    wf_base.figure_title = "2-panel plot with custom Base and Panel parameters"
    print("\nIn wf_panel, we can change the following parameters:")
    print("\t- units: str, units of the waveform.  Default 'Norm'")
    print("\t- label: str, label of the waveform.  Default '(wf)'")
    print("\t- yscaling: str, scaling of the y-axis.  Default 'auto'")
    print("\t- ytick_style: str, style of the y-axis ticks.  Default 'plain'")
    print("We'll change the units to 'Normalized'.")
    # set wf_panel optional parameters
    wf_panel.units = "Normalized"
    print("\nIn mesh_base, we can change the following parameters:")
    print("\t- frequency_scaling: str, scaling of the frequency axis.  Default 'log'")
    print("\t- shading: str, shading of the mesh.  Default 'auto'")
    print("\t- frequency_hz_ymin: Optional[float], minimum frequency to plot.  Default None")
    print("\t- frequency_hz_ymax: Optional[float], maximum frequency to plot.  Default None")
    print("\t- colormap: Optional[str], colormap to use.  Default None")
    print("\t- units_frequency: str, units of the frequency axis.  Default 'Hz'")
    print("We'll change the frequency limits to fmin and fmax, and the colormap to 'inferno.'")
    # set mesh_base optional parameters
    mesh_base.frequency_hz_ymin = fmin
    mesh_base.frequency_hz_ymax = fmax
    mesh_base.colormap = "inferno"
    print("\nIn mesh_panel, we can change the following parameters:")
    print("\t- colormap_scaling: str, scaling of the colormap.  Default 'auto'")
    print("\t- color_max: float, maximum value of the colormap.  Default None")
    print("\t- color_range: float, range of the colormap.  Default None")
    print("\t- color_min: float, minimum value of the colormap.  Default None")
    print("\t- cbar_units: str, units of the colorbar.  Default 'bits'")
    print("\t- ytick_style: str, style of the y-axis ticks.  Default 'sci'")
    print("\t- panel_label_color: str, color of the panel label.  Default 'black'")
    print("We'll change the colormap scaling to 'range', the range to 15, and the colorbar units to log2(Power)")
    mesh_panel.colormap_scaling = "range"
    mesh_panel.color_range = 15.
    mesh_panel.cbar_units = "log$_2$(Power)"
    print("\nLet's plot the customized figure now to see the changes. Close the figure when you're ready to continue.")
    # plot example figure
    stft = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    print("\n\t\t\t PLOTTING EXAMPLE FIGURE: CUSTOMIZED BASE AND PANEL PARAMETERS")
    plt.show()
    print("\nSince we changed the mesh colormap and the color scaling, we've created a new problem for ourselves: the")
    print("panel label disappears into the dark background of panel (a). We can fix this by changing the panel label")
    print("color parameter in mesh_panel to 'white'.")
    # set mesh_panel optional parameters
    mesh_panel.panel_label_color = "white"
    print("Let's plot the figure with the updated parameters. Close the figure when you're ready to continue.")
    stft = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    print("\n\t\t\t PLOTTING EXAMPLE FIGURE: CUSTOMIZED BASE AND PANEL PARAMETERS (UPDATED)")
    plt.show()
    # # # EXAMPLE: ADDING MORE MESH PANELS # # #
    print("\nNow that we've explored the Base and Panel objects, let's look at plot_n_mesh_wf_vert's parameters.")
    print("The required parameters are:")
    print("\t- mesh_base: MeshBase, base object for the mesh plot(s)")
    print("\t- mesh_panels: List[MeshPanel], list of mesh panels to plot")
    print("\t- waveform_base: WaveformPlotBase, base object for the waveform plot")
    print("\t- waveform_panel: WaveformPanel, waveform panel to plot")
    print("And the optional parameters are:")
    print("\t- sanitize_times: bool, if True, sanitize the time values.  Default True")
    print("\t- use_default_size: bool, if True, use the default figure size, otherwise size dynamically.  Default True")
    print("\nLet's see what happens if we add another mesh panel to the plot. Close the figures when you're ready to "
          "continue.")
    # create a second mesh panel
    [_, _, mic_stft_bits_2, _, _] = \
        calculate_tf_domain_example_values(mic_sig, time_fft_nd, freq_sample_rate_hz, freq_resolution_fft_hz, alpha=0.5)
    mesh_panel_2 = ptb.MeshPanel(tfr=mic_stft_bits_2, colormap_scaling="range", cbar_units="log$_2$(Power)",
                                 color_range=15., panel_label_color="white")
    # update the figure title
    wf_base.figure_title = "2-panel plot with default function parameters"
    stft_2panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    wf_base.figure_title = "3-panel plot with default function parameters"
    stft_3panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel, mesh_panel_2], wf_base, wf_panel)
    print("\n\t\t\t PLOTTING EXAMPLE FIGURES: ADDING MORE MESH PANELS\n")
    plt.show()
    print("By default, the figure size is maintained between plots. If instead, we'd like to maintain the panel size,")
    print("we can set use_default_size to False. Let's see what that looks like. Close the figures when you're ready to"
          " continue.")
    wf_base.figure_title = "2-panel plot with use_default_size = True"
    stft_2panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    wf_base.figure_title = "3-panel plot with use_default_size = True"
    stft_3panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel, mesh_panel_2], wf_base, wf_panel)
    wf_base.figure_title = "3-panel plot with use_default_size = False"
    stft_3panel2 = plot_n_mesh_wf_vert(mesh_base, [mesh_panel, mesh_panel_2], wf_base, wf_panel, use_default_size=False)
    print("\n\t\t\t PLOTTING EXAMPLE FIGURES: ADDING MORE MESH PANELS (MAINTAIN PANEL SIZE)\n")
    plt.show()
    print("Changing the figure size is also possible, if desired. We can set the figure size in the WaveformPlotBase ")
    print("object by setting the params_tfr input manually and giving it 1 of 5 built-in aspect ratios. We'll plot all")
    print("5 aspect ratios to see the difference. Close the figures when you're ready to continue.")
    wf_base.params_tfr = AudioParams(AspectRatioType(1))
    wf_base.figure_title = "2-panel plot with use_default_size = True and built-in figure size 1"
    stft_2panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    wf_base.params_tfr = AudioParams(AspectRatioType(2))
    wf_base.figure_title = "2-panel plot with use_default_size = True and built-in figure size 2"
    stft_2panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    wf_base.params_tfr = AudioParams(AspectRatioType(3))
    wf_base.figure_title = "2-panel plot with use_default_size = True and built-in figure size 3"
    stft_2panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    wf_base.params_tfr = AudioParams(AspectRatioType(4))
    wf_base.figure_title = "2-panel plot with use_default_size = True and built-in figure size 4"
    stft_2panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    wf_base.params_tfr = AudioParams(AspectRatioType(5))
    wf_base.figure_title = "2-panel plot with use_default_size = True and built-in figure size 5"
    stft_2panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    print("\n\t\t\t PLOTTING EXAMPLE FIGURES: ADJUSTING DEFAULT FIGURE SIZE\n")
    plt.show()
    print("In some cases, you may want to adjust the font size. This can be done by setting the text_size attribute")
    print("of AudioParams. To illustrate, we'll set the aspect ratio to AspectRatioType(5) and text_size to 12pt.")
    audio_params = AudioParams(AspectRatioType(5))
    audio_params.text_size = 12
    wf_base.params_tfr = audio_params
    wf_base.figure_title = "2-panel plot with built-in figure size 5 and 12pt font"
    stft_2panel = plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel)
    print("\n\t\t\t PLOTTING EXAMPLE FIGURE: ADJUSTING DEFAULT FIGURE SIZE AND FONT SIZE\n")
    plt.show()
    print("Finally, we'll demonstrate a functionalized version of the vertical plot template that is less customizable")
    print("but some users may find convenient. The functions plot_wf_mesh_mesh_vert_example and plot_wf_mesh_vert")
    print("example can be used to create identical plots to those demonstrated above. The following code demonstrates")
    print("the use of these functions. Additional parameters can be found in the documentation.")
    stft_2panel = plot_wf_mesh_vert_example(station_id="",
                                            wf_panel_a_sig=mic_sig,
                                            wf_panel_a_time=time_s,
                                            mesh_time=time_stft_s,
                                            mesh_frequency=frequency_stft_hz,
                                            mesh_panel_b_tfr=mic_stft_bits,
                                            frequency_hz_ymin=fmin,
                                            frequency_hz_ymax=fmax,
                                            mesh_panel_b_cbar_units="log$_2$(Power)",
                                            figure_title="2-panel plot created by functionalized plot template")
    stft_3panel = plot_wf_mesh_mesh_vert_example(station_id="",
                                                 wf_panel_a_sig=mic_sig,
                                                 wf_panel_a_time=time_s,
                                                 mesh_time=time_stft_s,
                                                 mesh_frequency=frequency_stft_hz,
                                                 mesh_panel_b_tfr=mic_stft_bits,
                                                 mesh_panel_c_tfr=mic_stft_bits_2,
                                                 frequency_hz_ymin=fmin,
                                                 frequency_hz_ymax=fmax,
                                                 mesh_panel_b_cbar_units="log$_2$(Power)",
                                                 mesh_panel_c_cbar_units="log$_2$(Power)",
                                                 figure_title="3-panel plot created by functionalized plot template")
    print("\n\t\t\t PLOTTING EXAMPLE FIGURE: FUNCTIONALIZED PLOT TEMPLATES\n")
    plt.show()
    print("This concludes the tutorial on the quantum-inferno vertical waveform and spectrogram plotting functions.")


if __name__ == "__main__":
    main()
