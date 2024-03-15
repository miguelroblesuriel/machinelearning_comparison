from matplotlib import pyplot as plt

def plot_scores(scores):
    #scores_array = scores.scores
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(scores.astype(float), cmap="viridis")
    plt.colorbar(shrink=0.7)
    plt.title("Cosine Greedy spectra similarities")
    plt.xlabel("Spectrum #ID")
    plt.ylabel("Spectrum #ID")
    plt.show()