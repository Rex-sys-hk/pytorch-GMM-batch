from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import time
import math
import torch
from torch import detach, nn
from torch import optim
import torch.distributions as D
from sklearn.datasets import make_blobs
import sklearn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def euclidean_metric_np(X, centroids):
    X = np.expand_dims(X, 1)
    centroids = np.expand_dims(centroids, 0)
    dists = (X - centroids) ** 2
    dists = np.sum(dists, axis=2)
    return dists


def euclidean_metric_gpu(X, centers):
    X = X.unsqueeze(1)
    centers = centers.unsqueeze(0)

    dist = torch.sum((X - centers) ** 2, dim=-1)
    return dist


def kmeans_fun_gpu(X, K=10, max_iter=1000, batch_size=8096, tol=1e-40):
    N = X.shape[0]

    indices = torch.randperm(N)[:K]
    init_centers = X[indices]

    batchs = N // batch_size
    last = 1 if N % batch_size != 0 else 0

    choice_cluster = torch.zeros([N]).to(device)
    for _ in range(max_iter):
        for bn in range(batchs + last):
            if bn == batchs and last == 1:
                _end = -1
            else:
                _end = (bn + 1) * batch_size
            X_batch = X[bn * batch_size: _end]

            dis_batch = euclidean_metric_gpu(X_batch, init_centers)
            choice_cluster[bn *
                           batch_size: _end] = torch.argmin(dis_batch, dim=1)

        init_centers_pre = init_centers.clone()
        for index in range(K):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
            selected = torch.index_select(X, 0, selected)
            init_centers[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((init_centers - init_centers_pre) ** 2, dim=1)
            ))
        if center_shift < tol:
            break

    k_mean = init_centers.detach().cpu().numpy()
    choice_cluster = choice_cluster.detach().cpu().numpy()
    return k_mean, choice_cluster


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=None, ax=None):
    ax = ax or plt.gca()
    if label is not None:
        ax.scatter(X[:, 0], X[:, 1], c=label, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weight.max()
    w_factor = w_factor.detach().cpu().numpy()
    for pos, covar, w in zip(gmm.means.detach().cpu().numpy(), gmm.std.detach().cpu().numpy(), gmm.weight.detach().cpu().numpy()):
        draw_ellipse(pos, covar, alpha=w * w_factor)


class GMM(nn.Module):

    def __init__(self, weights, stds,  n_comp=3, dim=2, means:torch.Tensor=None):
        super(GMM, self).__init__()
        self.weight_base = nn.Parameter(weights)
        self.weight = 0
        if means is None:
            means = torch.rand([n_comp, dim]).to(device)
        else:
            means = means.to(device)
        self.means = nn.Parameter(means)

        self.std = nn.Parameter(stds)
        self.dim = dim
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, w: torch.Tensor = None):
        if w is None:
            w = torch.ones(x.shape[0]).to(device)

        self.weight = self.soft(self.weight_base)
        mix = D.Categorical(self.weight)
        mvn = D.MultivariateNormal(self.means, self.std)
        comp = D.Independent(mvn, 0)
        gmm = D.MixtureSameFamily(mix, comp)
        return (-gmm.log_prob(x)*w).mean()


comp_n = 9
dim = 2
N = 100
X, _ = make_blobs(n_samples=N, n_features=dim, centers=comp_n)
X = torch.tensor(X).to(device)
last_loss = torch.tensor(math.inf).to(device)

if __name__ == "__main__":
    weight = torch.ones(comp_n).to(device)
    st = time.time()
    mean, pre_label = kmeans_fun_gpu(X, K=comp_n, max_iter=1000, tol=1e-4)
    print('k-means cost time:', time.time()-st)
    mean = torch.tensor(mean).to(device)
    stds = torch.eye(dim).repeat(comp_n, 1, 1).to(device)
    model = GMM(weight, stds, n_comp=comp_n, means = mean).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.2)
    # optimizer = optim.Adam(model.parameters(), lr=1, momentum=0.2)

    start = time.time()
    for i in range(100):
        optimizer.zero_grad()
        loss = model(X)
        loss.backward()
        optimizer.step()

        print('epoch:', i, 'loss is:', loss.item())
        if last_loss-loss < 1e-3:
            break
        last_loss = loss.item()
    print("training time", time.time()-start)

    print(X.shape)
    plt.scatter(X.cpu().detach().numpy()[:, 0], X.cpu().detach().numpy()[:, 1])
    print("means shape:", model.means.shape)
    plt.scatter(model.means.cpu().detach().numpy()[
                :, 0], model.means.cpu().detach().numpy()[:, 1])
    plot_gmm(gmm=model,X = X, label=pre_label)
    plt.show()
