from time import time

import datafold
import megaman.embedding
import numpy as np
from megaman.geometry import Geometry
from sklearn import manifold, datasets
from tqdm import tqdm
import pickle


def main():
    num_samples = 10000
    X = np.load('./news_subset.npy')[:num_samples]
    n_components = 2

    print('\nRunning on Google news subset dataset')
    embeddings = process_dataset(X, n_components, geom_radius=20.0)
    with open('google_news_embedding.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    X, color = datasets.make_s_curve(num_samples, random_state=0)
    print('\nRunning on sklearn S curve dataset')
    embeddings = process_dataset(X, n_components, geom_radius=0.09)
    with open('s_curve_embedding.pkl', 'wb') as f:
        pickle.dump(embeddings, f)


def process_dataset(X, n_components, geom_radius=1.0):
    print(f'dataset shape: {X.shape}')

    megaman_algos = get_megaman_algorithms(n_components, geom_radius)
    sklearn_algos = get_sklearn_algorithms(n_components)
    datafold_algos = {'datafold_diffusion_maps': datafold.dynfold.DiffusionMaps(n_eigenpairs=n_components)}

    all_algos = {**megaman_algos, **sklearn_algos, **datafold_algos}
    embeddings, timings = run_fit_transform(X, all_algos)
    metrics = compute_metrics(X, embeddings)

    print(f"\nTimings: {timings}")
    print(f"Metrics: {metrics}")
    return embeddings


def compute_metrics(X, embeddings):
    metrics = {}
    for name, embedding in tqdm(embeddings.items(), desc='Computing Metrics'):
        metrics[name] = manifold.trustworthiness(X, embedding)
    return metrics


def run_fit_transform(X, all_algos):
    embeddings = {}
    timings = {}
    pbar = tqdm(all_algos.items(), desc='Running fit_transform')
    for name, algo in pbar:
        pbar.set_description(f'Running {name}')
        start_time = time()
        embeddings[name] = algo.fit_transform(X)
        timings[name] = time() - start_time

    return embeddings, timings


def get_megaman_algorithms(n_components, geom_radius):
    geom = get_geometry(radius=geom_radius)
    spectral = megaman.embedding.SpectralEmbedding(n_components=n_components, eigen_solver='amg', geom=geom,
                                                   drop_first=False)
    lle = megaman.embedding.LocallyLinearEmbedding(n_components=n_components, eigen_solver='arpack', geom=geom)
    isomap = megaman.embedding.Isomap(n_components=n_components, eigen_solver='arpack', geom=geom)
    return {
        'megaman_spectral': spectral,
        'megaman_lle': lle,
        'megaman_isomap': isomap,
    }


def get_sklearn_algorithms(n_components):
    spectral = manifold.SpectralEmbedding(n_components=n_components, eigen_solver='amg')
    lle = manifold.LocallyLinearEmbedding(n_components=n_components, eigen_solver='arpack')
    isomap = manifold.Isomap(n_components=n_components, eigen_solver='arpack')
    return {
        'sklearn_spectral': spectral,
        'sklearn_lle': lle,
        'sklearn_isomap': isomap,
    }


def get_geometry(radius):
    adjacency_method = 'cyflann'
    adjacency_kwds = {'radius': radius}
    affinity_method = 'gaussian'
    affinity_kwds = {'radius': radius}
    laplacian_method = 'symmetricnormalized'
    laplacian_kwds = {'scaling_epps': radius}
    geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                    affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                    laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
    return geom


if __name__ == '__main__':
    main()
