from matplotlib import pyplot as plt

def mirror_plot(spectrum1, spectrum2, i=None, precmz1=None, precmz2=None, rt1=None, rt2=None):
    plt.figure(figsize=(8, 4))
    # Plot Spectrum 1
    plt.stem(spectrum1.mz, spectrum1.intensities, linefmt='b-', markerfmt='', basefmt=' ',
             label=f'Spectrum 1- precmz={precmz1} - rt={rt1}')

    # Plot Spectrum 2 with reversed m/z values
    plt.stem(spectrum2.mz, -spectrum2.intensities, linefmt='r-', markerfmt='', basefmt=' ',
             label=f'Spectrum 2 (Mirrored) - precmz={precmz2} - rt={rt2}')
    # Add labels and legend
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title('Mirror Plot Along X-axis of MS2 Spectra')
    plt.legend()

    # Show the mirror plot
    plt.savefig("mirror " + str(i) + ".png", dpi=300, bbox_inches="tight", transparent=True)