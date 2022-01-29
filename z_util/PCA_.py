from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('seaborn')
sns.set_theme(font_scale =1.5)

def apply_pca(X, components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=components)
    Xp = pca.fit_transform(X_scaled)
    return Xp

def plot_pca(X, Xp):
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.title('X1 vs. X0')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.scatter(X[:, 0], X[:, 1])

    plt.subplot(1, 2, 2)
    plt.title('PC1 vs PC2')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.scatter(Xp[:, 0], Xp[:, 1])
