import enum
from typing import Union
import numpy as np
from sklearn.decomposition import PCA, NMF
import umap

from xicam.core.intents import ImageIntent, PlotIntent, BarIntent, PairPlotIntent
from xicam.plugins.operationplugin import operation, categories, display_name, output_names, visible, input_names, \
    intent


class svd_solver(enum.Enum):
    auto = 'auto'
    full = 'full'
    arpack = 'arpack'
    randomized = 'randomized'


@operation
@categories('Decomposition')
# @categories(('Spectral Imaging', 'Decomposition'), ('BSISB', 'Decomposition'))
@display_name('PCA (Principal Component Analysis')
@output_names('transform_data',
              'variance',
              'components',
              # *[f'component{i}' for i in range(3)],
              'energy', 'component_indices')
@visible('data', False)
# @input_names('data', 'energy axis is last', 'Number of components', 'Copy', 'Whiten', 'SVD Solver', 'Tolerance', 'Iterated Power', 'Random State')
@intent(PairPlotIntent, "Pair Plot", output_map={"transform_data": "transform_data"})
@intent(ImageIntent, "PCA Transform", output_map={"image": "transform_data", 'xvals': 'component_indices'}, mixins=['SliceSelector'], axes={'t': 2, 'x': 1, 'y': 0, 'c': None})
@intent(ImageIntent, "False-color PCA Transform", output_map={"image": "transform_data", 'xvals': 'component_indices'}, axes={'t': None, 'x': 1, 'y': 0, 'c': 2})
@intent(BarIntent, "Variance Ratio", x=[1,2,3], width=1, output_map={"height": "variance"}, labels={'left': 'Variance Ratio'})
@intent(PlotIntent, "Component", match_key='components', output_map={"y": "components", 'x': 'energy'}, labels={'left': 'PCA 1', 'bottom': 'Energy (eV)'})
# @intent(PlotIntent, "Component 1", match_key='components', output_map={"y": "component1", 'x': 'energy'}, labels={'left': 'PCA 2', 'bottom': 'Energy (eV)'})
# @intent(PlotIntent, "Component 2", match_key='components', output_map={"y": "component2", 'x': 'energy'}, labels={'left': 'PCA 3', 'bottom': 'Energy (eV)'})
def pca(data:np.ndarray, n_components:Union[int, float, str]=3, copy:bool=True, whiten:bool=False, svd_solver:svd_solver='auto', tol:float=0.0, iterated_power:Union[str, int]='auto', random_state:int=None):
    pca = PCA(n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state)

    pca.fit(np.asarray(data).reshape(-1, data.shape[0]))
    images = pca.transform(np.asarray(data).reshape(-1, data.shape[0])).reshape(data.shape[1], data.shape[2], 3)
    return images, pca.explained_variance_ratio_, pca.components_, data.coords['E (eV)'], np.arange(images.shape[-1])


class init(enum.Enum):
    random='random'
    nndsvda='nndsvda'
    nndsvdar='nndsvdar'
    custom='custom'


class solver(enum.Enum):
    cd='cd'
    mu='mu'


class beta_loss(enum.Enum):
    frobenius='frobenius'
    kullback_leibler='kullback-leibler'
    itakura_saito='itakura-saito'


@operation
@categories('Decomposition')
# @categories(('Spectral Imaging', 'Decomposition'), ('BSISB', 'Decomposition'))
@display_name('NMF (Non-negative Matrix Factorization')
@output_names('data')
@visible('data', False)
@input_names(None, 'Number of components', 'Initialization', 'Solver', 'Beta Loss', 'Tolerance', 'Maximum Iterations', 'Random State', 'Alpha', 'Shuffle')
def nmf(data:np.ndarray, n_components:Union[int, float, str]=None, init:init=None, solver:solver=solver.cd, beta_loss:beta_loss=beta_loss.frobenius, tol:float=1e-4, max_iter:int=200, random_state:int=None, alpha:float=0, shuffle:bool=False) -> np.ndarray:
    nmf = NMF(n_components, init=init, solver=solver, beta_loss=beta_loss, tol=tol, max_iter=max_iter, random_state=random_state, alpha=alpha, shuffle=shuffle)
    return nmf.fit_transform(data)


class metric(enum.Enum):
    euclidean = 'euclidean'
    manhattan = 'manhattan'
    chebyshev = 'chebyshev'
    minkowski = 'minkowski'
    canberra = 'canberra'
    braycurtis = 'braycurtis'
    mahalanobis = 'mahalanobis'
    wminkowski = 'wminkowski'
    seuclidean = 'seuclidean'
    cosine = 'cosine'
    correlation = 'correlation'
    haversine = 'haversine'
    hamming = 'hamming'
    jaccard = 'jaccard'
    dice = 'dice'
    russelrao = 'russelrao'
    kulsinski = 'kulsinki'
    ll_dirichlet = 'll_dirichlet'
    hellinger = 'hellinger'
    rogerstanimoto = 'rogerstanimoto'
    sokalmichener = 'sokalmichener'
    sokalsneath = 'sokalsneath'
    yule = 'yule'


@operation
# @categories(('Spectral Imaging', 'Decomposition'), ('BSISB', 'Decomposition'))
@categories('Decomposition')
@display_name('Uniform Manifold Approximation and Projection')
@output_names('data')
@visible('data', False)
@input_names(None, 'Number of neighbors', 'Number of components', 'Distance metric', None)
def umap(data:np.ndarray, n_neighbors:int=15, n_components:int=2, metric:metric='euclidean', **kwargs):#metric_kwds=None, output_metric='euclidean', output_metric_kwds=None, n_epochs=None, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, low_memory=False, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, force_approximation_algorithm=False, unique=False):
    return umap.UMAP(n_neighbors, n_components, metric, **kwargs).fit_transform(data)#metric_kwds, output_metric, output_metric_kwds, n_epochs, learning_rate, init, min_dist, spread, low_memory, set_op_mix_ratio, local_connectivity, repulsion_strength, negative_sample_rate, transform_queue_size, a, b, random_state, angular_rp_forest, target_n_neighbors, target_metric, target_metric_kwds, target_weight, transform_seed, force_approximation_algorithm, unique).fit_transform(data)

# TODO: split into fit and transform operations

# PCA
## https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# NMF
## https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF
# UMAP
## https://github.com/lmcinnes/umap
