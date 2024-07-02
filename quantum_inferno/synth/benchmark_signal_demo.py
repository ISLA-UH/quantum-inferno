"""
Demonstrate the synthetic benchmark signals.
"""
import matplotlib.pyplot as plt

import quantum_inferno.synth.benchmark_signals as bsig


if __name__ == "__main__":
    sig_wf, sig_t = bsig.synth_00()
    plt.figure()
    plt.plot(sig_t, sig_wf)
    plt.title("Synth 00")

    sig_wf, sig_t = bsig.synth_01()
    plt.figure()
    plt.plot(sig_t, sig_wf)
    plt.title("Synth 01")

    sig_wf, sig_t = bsig.synth_02()
    plt.figure()
    plt.plot(sig_t, sig_wf)
    plt.title("Synth 02")

    sig_wf, sig_t = bsig.synth_03()
    plt.figure()
    plt.plot(sig_t, sig_wf)
    plt.title("Synth 03")

    plt.show()
