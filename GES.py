from typing import Optional
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag
from typing import Union

from scipy.linalg import sqrtm, inv
from scipy.spatial.distance import cdist
from numpy.random import default_rng


def kernel_midwidth_rbf(X):
    n = len(X)
    # ----- width of X -----
    Xmed = X

    G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * (Xmed @ Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))

    return width_x


def rbf_midwidth_multi(Data, parameters, max_size=500):
    Data0 = Data[:max_size, :]
    node_num = len(parameters["dlabel"])
    width_all = []
    for i in range(node_num):
        indx = parameters["dlabel"][i]
        if len(indx) == 1:
            X = Data[:, indx].reshape(-1, 1)
        else:
            X = Data[:, indx]
        width_x = kernel_midwidth_rbf(X)
        width_all.append(width_x)
    return width_all


def rbf_midwidth_general(Data, max_size=500):
    Data0 = Data[:max_size, :]
    node_num = Data0.shape[1]
    width_all = []
    for i in range(node_num):
        X = Data0[:, i].reshape(-1, 1)
        width_x = kernel_midwidth_rbf(X)
        width_all.append(width_x)
    return width_all


def ges(X: ndarray, score_func: str = 'local_score_BIC', maxP: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None, node_names: Union[List[str], None] = None, ) -> Dict[str, Any]:
    """
    Perform greedy equivalence search (GES) algorithm

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
                    'local_score_CV_multi', 'local_score_marginal_multi', 'local_score_BIC', 'local_score_BDeu')).
    maxP : allowed maximum number of parents when searching the graph
    parameters : when using CV likelihood,
                  parameters['kfold']: k-fold cross validation
                  parameters['lambda']: regularization parameter
                  parameters['dlabel']: for variables with multi-dimensions,
                               indicate which dimensions belong to the i-th variable.

    Returns
    -------
    Record['G']: learned causal graph, where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicates  i --> j ,
                    Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.
    Record['update1']: each update (Insert operator) in the forward step
    Record['update2']: each update (Delete operator) in the backward step
    Record['G_step1']: learned graph at each step in the forward step
    Record['G_step2']: learned graph at each step in the backward step
    Record['score']: the score of the learned graph
    """

    if X.shape[0] < X.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")
    maxP = X.shape[1] - 1
    threshold = 0.5
    if 'nym' in score_func:
        X = np.array(X)
    else:
        X = np.mat(X)
    if score_func == 'local_score_CV_general':  # % k-fold negative cross validated likelihood based on regression in RKHS
        if parameters is None:
            parameters = {'kfold': 10,  # 10 fold cross validation
                          'lambda': 0.01}  # regularization parameter
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_general, parameters=parameters)

    elif score_func == 'local_score_marginal_general':  # negative marginal likelihood based on regression in RKHS
        parameters = {}
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_general, parameters=parameters)

    elif score_func == 'local_score_CV_multi':  # k-fold negative cross validated likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'kfold': 10, 'lambda': 0.01, 'dlabel': {}}  # regularization parameter
            for i in range(X.shape[1]):
                parameters['dlabel'][i] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_multi, parameters=parameters)

    elif score_func == 'local_score_marginal_multi':  # negative marginal likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'dlabel': {}}
            for i in range(X.shape[1]):
                parameters['dlabel'][i] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)

    elif score_func == 'local_score_BIC' or score_func == 'local_score_BIC_from_cov':  # Greedy equivalence search with BIC score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        parameters = {}
        parameters["lambda_value"] = 2
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters)

    elif score_func == 'local_score_BDeu':  # Greedy equivalence search with BDeu score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)

    elif score_func == 'local_score_cv_general_nym':
        if parameters is None:
            parameters = {'kfold': 10,  # 10 fold cross validation
                          'lambda': 0.01}  # regularization parameter
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        MidWidth = rbf_midwidth_general(X, max_size=500)
        parameters['MidWidth'] = MidWidth
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_general_nym, parameters=parameters)

    elif score_func == 'local_score_cv_multi_nym':
        N = len(parameters['dlabel'])
        maxP = N - 1  # maximum number of parents
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_multi_nym, parameters=parameters)
    elif score_func == 'local_score_cv_discrete_nym':
        if parameters is None:
            parameters = {'kfold': 10,  # 10 fold cross validation
                          'lambda': 0.01}  # regularization parameter
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        MidWidth = rbf_midwidth_general(X, max_size=500)
        parameters['MidWidth'] = MidWidth
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_discrete_nym, parameters=parameters)


    else:
        raise Exception('Unknown function!')
    score_func = localScoreClass

    if node_names is None:
        node_names = [("X%d" % (i + 1)) for i in range(N)]
    nodes = []

    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)

    G = GeneralGraph(nodes)
    # G = np.matlib.zeros((N, N)) # initialize the graph structure
    score = score_g(X, G, score_func, parameters)  # initialize the score

    G = pdag2dag(G)
    G = dag2cpdag(G)

    ## --------------------------------------------------------------------
    ## forward greedy search
    record_local_score = [[] for i in range(
        N)]  # record the local score calculated each time. Thus when we transition to the second phase,
    # many of the operators can be scored without an explicit call the the scoring function
    # record_local_score{trial}{j} record the local scores when Xj as a parent
    score_new = score
    count1 = 0
    update1 = []
    G_step1 = []
    score_record1 = []
    graph_record1 = []
    while True:
        count1 = count1 + 1
        score = score_new
        score_record1.append(score)
        graph_record1.append(G)
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                if (G.graph[i, j] == Endpoint.NULL.value and G.graph[j, i] == Endpoint.NULL.value
                        and i != j and len(np.where(G.graph[j, :] == Endpoint.ARROW.value)[0]) <= maxP):
                    # find a pair (Xi, Xj) that is not adjacent in the current graph , and restrict the number of parents
                    Tj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj

                    Ti = np.union1d(np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
                                    np.where(G.graph[i, :] != Endpoint.NULL.value)[0])  # adjacent to Xi

                    NTi = np.setdiff1d(np.arange(N), Ti)
                    T0 = np.intersect1d(Tj, NTi)  # find the neighbours of Xj that are not adjacent to Xi
                    # for any subset of T0
                    sub = Combinatorial(T0.tolist())  # find all the subsets for T0
                    S = np.zeros(len(sub))
                    # S indicate whether we need to check sub{k}.
                    # 0: check both conditions.
                    # 1: only check the first condition
                    # 2: check nothing and is not valid.
                    for k in range(len(sub)):
                        if (S[k] < 2):  # S indicate whether we need to check subset(k)
                            V1 = insert_validity_test1(G, i, j, sub[k])  # Insert operator validation test:condition 1
                            if (V1):
                                if (not S[k]):
                                    V2 = insert_validity_test2(G, i, j,
                                                               sub[k])  # Insert operator validation test:condition 2
                                else:
                                    V2 = 1
                                if (V2):
                                    Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
                                    S[np.where(Idx == 1)] = 1
                                    chscore, desc, record_local_score = insert_changed_score(X, G, i, j, sub[k],
                                                                                             record_local_score,
                                                                                             score_func,
                                                                                             parameters)
                                    # calculate the changed score after Insert operator
                                    # desc{count} saves the corresponding (i,j,sub{k})
                                    # sub{k}:
                                    if (chscore < min_chscore):
                                        min_chscore = chscore
                                        min_desc = desc
                            else:
                                Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
                                S[np.where(Idx == 1)] = 2

        if (len(min_desc) != 0):
            score_new = score + min_chscore
            if (score - score_new <= threshold):
                break
            G = insert(G, min_desc[0], min_desc[1], min_desc[2])
            update1.append([min_desc[0], min_desc[1], min_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step1.append(G)
        else:
            score_new = score
            break

    ## --------------------------------------------------------------------
    # backward greedy search
    count2 = 0
    score_new = score
    update2 = []
    G_step2 = []
    score_record2 = []
    graph_record2 = []
    while True:
        count2 = count2 + 1
        score = score_new
        score_record2.append(score)
        graph_record2.append(G)
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                if ((G.graph[j, i] == Endpoint.TAIL.value and G.graph[i, j] == Endpoint.TAIL.value)
                        or G.graph[j, i] == Endpoint.ARROW.value):  # if Xi - Xj or Xi -> Xj
                    Hj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
                    Hi = np.union1d(np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                                    np.where(G.graph[:, i] != Endpoint.NULL.value)[0])  # adjacent to Xi
                    H0 = np.intersect1d(Hj, Hi)  # find the neighbours of Xj that are adjacent to Xi
                    # for any subset of H0
                    sub = Combinatorial(H0.tolist())  # find all the subsets for H0
                    S = np.ones(len(sub))  # S indicate whether we need to check sub{k}.
                    # 1: check the condition,
                    # 2: check nothing and is valid;
                    for k in range(len(sub)):
                        if (S[k] == 1):
                            V = delete_validity_test(G, i, j, sub[k])  # Delete operator validation test
                            if (V):
                                # find those subsets that include sub(k)
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2  # and set their S to 2
                        else:
                            V = 1

                        if (V):
                            chscore, desc, record_local_score = delete_changed_score(X, G, i, j, sub[k],
                                                                                     record_local_score, score_func,
                                                                                     parameters)
                            # calculate the changed score after Insert operator
                            # desc{count} saves the corresponding (i,j,sub{k})
                            if (chscore < min_chscore):
                                min_chscore = chscore
                                min_desc = desc

        if len(min_desc) != 0:
            score_new = score + min_chscore
            if score - score_new <= threshold:
                break
            G = delete(G, min_desc[0], min_desc[1], min_desc[2])
            update2.append([min_desc[0], min_desc[1], min_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step2.append(G)
        else:
            score_new = score
            break

    Record = {'update1': update1, 'update2': update2, 'G_step1': G_step1, 'G_step2': G_step2, 'G': G, 'score': score}
    return Record


def pdinv_frac(A):
    # PDINV Computes the inverse of a positive definite matrix
    numData = A.shape[0]
    try:
        U = np.linalg.cholesky(A)
        invU = (np.eye(numData).dot(np.linalg.inv(U))).T
    except numpy.linalg.LinAlgError as e:
        warnings.warn('Matrix is not positive definite in pdinv, inverting using svd')
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        invU = vh.T.dot(np.diag(1 / np.sqrt(s)))
    except Exception as e:
        raise e
    return invU


def feat_gen(X, wx):
    fX = X / wx
    return fX


def pdinv_Array(A):
    # PDINV Computes the inverse of a positive definite matrix
    numData = A.shape[0]
    try:
        U = np.linalg.cholesky(A).T
        invU = np.eye(numData).dot(np.linalg.inv(U))
        Ainv = invU.dot(invU.T)
    except numpy.linalg.LinAlgError as e:
        warnings.warn('Matrix is not positive definite in pdinv, inverting using svd')
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        Ainv = vh.T.dot(np.diag(1 / s)).dot(u.T)
    except Exception as e:
        raise e
    # print(Ainv.shape)
    return Ainv


def local_score_cv_discrete_nym(
        Data: ndarray, Xi: int, PAi: List[int], parameters: Dict[str, Any]
) -> float:
    """
    Calculate the local score
    using negative k-fold cross-validated log likelihood as the score
    based on a regression model in RKHS

    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters:
                   kfold: k-fold cross validation
                   lambda: regularization parameter

    Returns
    -------
    score: local score
    """

    PAi = list(PAi)
    mpa = None
    T = Data.shape[0]
    X = Data[:, Xi].reshape(-1, 1)
    var_lambda = parameters["lambda"]  # regularization parameter
    k = parameters["kfold"]  # k-fold cross validation
    m = parameters["m"]
    MidWidth = parameters['MidWidth']
    # max_sample = parameters["max_sample"] # samples use to calculate the midwidth
    n0 = math.floor(T / k)
    gamma = 0.01
    beta = var_lambda ** 2 / gamma
    Thresh = 1e-5
    deltaa = 1e-7

    kerneld = DiagGaussianKernel()

    if len(PAi):
        PA = Data[:, PAi]
        if PA.shape[1] == 1:
            PA = PA.reshape(-1, 1)

        # induce matrix for X
        fX = feat_gen(X, MidWidth[Xi] * 2)
        inds = np.unique(fX).reshape(-1, 1)
        if m >= len(inds):
            mx = len(inds)
            Ks = np.exp(-0.5 * cdist(inds, inds, 'sqeuclidean'))
            invU = pdinv_frac(Ks)
            phi_X = np.exp(-0.5 * cdist(fX, inds, 'sqeuclidean')) @ invU
        else:
            # induce matrix for X
            fX = feat_gen(X, MidWidth[Xi] * 2)
            phi_X = kernel_icd(fX, kerneld, m)
            mx = phi_X.shape[1]

        # induce matrix for PA
        fPA = feat_gen(PA, np.array(MidWidth)[PAi] * 2)
        inds_pa = np.unique(fPA, axis=0)
        if m >= len(inds_pa):
            mpa = len(inds_pa)
            Ks_pa = np.exp(-0.5 * cdist(inds_pa, inds_pa, 'sqeuclidean'))
            invUpa = pdinv_frac(Ks_pa)
            phi_PA = np.exp(-0.5 * cdist(fPA, inds_pa, 'sqeuclidean')) @ invUpa
        else:
            # induce matrix for PA
            fPA = feat_gen(PA, np.array(MidWidth)[PAi] * 2)
            phi_PA = kernel_icd(fPA, kerneld, m)
            mpa = phi_PA.shape[1]

        # centering
        phi_Xc = phi_X - np.mean(phi_X, 0)
        phi_PAc = phi_PA - np.mean(phi_PA, 0)

        CV = 0
        for kk in range(k):
            if kk == 0:
                phi_Xc_te = phi_Xc[0:n0, :]
                phi_Xc_tr = phi_Xc[n0:T, :]
                phi_PAc_te = phi_PAc[0:n0, :]
                phi_PAc_tr = phi_PAc[n0:T, :]
                nv = n0  # sample size of validated data
            elif kk == k - 1:
                phi_Xc_te = phi_Xc[kk * n0: T, :]
                phi_Xc_tr = phi_Xc[0: kk * n0, :]
                phi_PAc_te = phi_PAc[kk * n0: T, :]
                phi_PAc_tr = phi_PAc[0: kk * n0, :]
                nv = T - kk * n0
            elif kk < k - 1 and kk > 0:
                idn = np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)])
                phi_Xc_te = phi_Xc[kk * n0: (kk + 1) * n0, :]
                phi_Xc_tr = phi_Xc[idn, :]
                phi_PAc_te = phi_PAc[kk * n0: (kk + 1) * n0, :]
                phi_PAc_tr = phi_PAc[idn, :]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            Pr = phi_Xc_tr.T @ phi_Xc_tr  # mx x mx
            Er = phi_PAc_tr.T @ phi_Xc_tr  # mpa x mx
            Fr = phi_PAc_tr.T @ phi_PAc_tr  # mpa x mpa
            Vr = phi_Xc_te.T @ phi_Xc_te  # mx x mx
            Ur = phi_PAc_te.T @ phi_Xc_te  # mpa x mx
            Sr = phi_PAc_te.T @ phi_PAc_te  # mpa x mpa

            Dr = pdinv_Array(n1 * var_lambda * np.eye(mpa) + Fr)  # mpa x mpa

            de = Dr @ Er  # mpa x mx
            T0 = Pr - 2 * Er.T @ de + de.T @ Fr @ de  # mx x mx
            Gr = pdinv_Array(np.eye(mx) + 1 / (n1 * gamma) * T0)  # mx x mx

            df = Dr @ Fr  # mpa x mpa
            twoi_df = 2 * np.eye(mpa) - df  # mpa x mpa

            kappa = (n1 * beta) / ((n1 * var_lambda) ** 4)

            Afrak = -1 / (n1 * var_lambda) ** 2 * (twoi_df @ Dr) - kappa * (twoi_df) @ de @ Gr @ de.T @ (twoi_df.T)
            Bfrak = kappa * (Gr @ de.T) @ (twoi_df.T)
            Cfrak = Bfrak.T
            Dfrak = -kappa * Gr

            # print(n1, Afrak.shape, Bfrak.shape, Cfrak.shape, Dfrak.shape, rff_Xc_tr.shape, rff_PAc_tr.shape)
            # Cr = 1/((n1*var_lambda)**2) * np.eye(n1) + (phi_PAc_tr@Afrak@phi_PAc_tr.T) + (phi_Xc_tr@Bfrak@phi_PAc_tr.T) + (phi_PAc_tr@Cfrak@phi_Xc_tr.T) + (phi_Xc_tr@Dfrak@phi_Xc_tr.T)

            i_df = np.eye(mpa) - df  # mpa x mpa
            Wr = 1 / ((
                              n1 * var_lambda) ** 2) * Pr + Er.T @ Afrak @ Er + Pr @ Dfrak @ Pr + Pr @ Bfrak @ Er + Er.T @ Cfrak @ Pr  # mx x mx

            # first part
            Qr = 1 / (n1 * gamma) * T0 + np.eye(mx)  # mx x mx
            Rr = np.linalg.cholesky(Qr)  # mx x mx
            first_part = np.sum(np.log(np.diag(Rr)))

            # second part
            etidf = Er.T @ i_df  # mx x mpa
            second_part = np.trace((np.eye(mx) - n1 * beta * Wr) @ (
                    Vr - 2 / (n1 * var_lambda) * etidf @ Ur + 1 / ((n1 * var_lambda) ** 2) * etidf @ Sr @ etidf.T))


            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * first_part + second_part / gamma) / 2

        CV = CV / k
    else:
        fX = feat_gen(X, MidWidth[Xi] * 2)
        inds = np.unique(fX).reshape(-1, 1)

        if m >= len(inds):
            mx = len(inds)
            Ks = np.exp(-0.5 * cdist(inds, inds, 'sqeuclidean'))
            invU = pdinv_frac(Ks)
            phi_X = np.exp(-0.5 * cdist(fX, inds, 'sqeuclidean')) @ invU
        else:
            # induce matrix for PA
            fX = feat_gen(X, MidWidth[Xi] * 2)
            phi_X = kernel_icd(fX, kerneld, m)
            mx = phi_X.shape[1]

        # centering
        phi_Xc = phi_X - np.mean(phi_X, 0)

        CV = 0
        for kk in range(k):
            if kk == 0:
                phi_Xc_te = phi_Xc[0:n0, :]
                phi_Xc_tr = phi_Xc[n0:T, :]
                nv = n0
            elif kk == k - 1:
                phi_Xc_te = phi_Xc[kk * n0: T, :]
                phi_Xc_tr = phi_Xc[0: kk * n0, :]
                nv = T - kk * n0
            elif 0 < kk < k - 1:
                idn = np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)])
                phi_Xc_te = phi_Xc[kk * n0: (kk + 1) * n0, :]
                phi_Xc_tr = phi_Xc[idn, :]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            Pr = phi_Xc_tr.T @ phi_Xc_tr
            Vr = phi_Xc_te.T @ phi_Xc_te

            Qr = 1 / (n1 * gamma) * Pr + np.eye(mx)
            Rr = np.linalg.cholesky(Qr)
            first_part = np.sum(np.log(np.diag(Rr)))

            Dr = pdinv_Array(np.eye(mx) + 1 / (n1 * gamma) * Pr)
            vp = Vr @ Pr
            second_part = np.trace(Vr - 1 / (n1 * gamma) * vp + (1 / (n1 * gamma) ** 2) * vp @ Dr @ Pr)


            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * first_part + second_part / gamma) / 2

        CV = CV / k

    score = CV  # negative cross-validated likelihood
    return score


def local_score_cv_general_nym(
        Data: ndarray, Xi: int, PAi: List[int], parameters: Dict[str, Any]
) -> float:
    """
    Calculate the local score
    using negative k-fold cross-validated log likelihood as the score
    based on a regression model in RKHS

    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters:
                   kfold: k-fold cross validation
                   lambda: regularization parameter

    Returns
    -------
    score: local score
    """

    PAi = list(PAi)
    mpa = None
    T = Data.shape[0]
    X = Data[:, Xi].reshape(-1, 1)
    var_lambda = parameters["lambda"]  # regularization parameter
    k = parameters["kfold"]  # k-fold cross validation
    m = parameters["m"]
    MidWidth = parameters['MidWidth']
    # max_sample = parameters["max_sample"] # samples use to calculate the midwidth
    n0 = math.floor(T / k)
    gamma = 0.01
    beta = var_lambda ** 2 / gamma
    Thresh = 1e-5
    deltaa = 1e-7

    kerneld = DiagGaussianKernel()

    if len(PAi):
        PA = Data[:, PAi]
        if PA.shape[1] == 1:
            PA = PA.reshape(-1, 1)

        # induce matrix for X
        fX = feat_gen(X, MidWidth[Xi] * 2)
        phi_X = kernel_icd(fX, kerneld, m)

        # induce matrix for PA
        fPA = feat_gen(PA, np.array(MidWidth)[PAi] * 2)
        phi_PA = kernel_icd(fPA, kerneld, m)

        mx = phi_X.shape[1]
        mpa = phi_PA.shape[1]

        # centering
        phi_Xc = phi_X - np.mean(phi_X, 0)
        phi_PAc = phi_PA - np.mean(phi_PA, 0)

        CV = 0
        for kk in range(k):
            if kk == 0:
                phi_Xc_te = phi_Xc[0:n0, :]
                phi_Xc_tr = phi_Xc[n0:T, :]
                phi_PAc_te = phi_PAc[0:n0, :]
                phi_PAc_tr = phi_PAc[n0:T, :]
                nv = n0  # sample size of validated data
            elif kk == k - 1:
                phi_Xc_te = phi_Xc[kk * n0: T, :]
                phi_Xc_tr = phi_Xc[0: kk * n0, :]
                phi_PAc_te = phi_PAc[kk * n0: T, :]
                phi_PAc_tr = phi_PAc[0: kk * n0, :]
                nv = T - kk * n0
            elif kk < k - 1 and kk > 0:
                idn = np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)])
                phi_Xc_te = phi_Xc[kk * n0: (kk + 1) * n0, :]
                phi_Xc_tr = phi_Xc[idn, :]
                phi_PAc_te = phi_PAc[kk * n0: (kk + 1) * n0, :]
                phi_PAc_tr = phi_PAc[idn, :]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            Pr = phi_Xc_tr.T @ phi_Xc_tr  # mx x mx
            Er = phi_PAc_tr.T @ phi_Xc_tr  # mpa x mx
            Fr = phi_PAc_tr.T @ phi_PAc_tr  # mpa x mpa
            Vr = phi_Xc_te.T @ phi_Xc_te  # mx x mx
            Ur = phi_PAc_te.T @ phi_Xc_te  # mpa x mx
            Sr = phi_PAc_te.T @ phi_PAc_te  # mpa x mpa

            Dr = pdinv_Array(n1 * var_lambda * np.eye(mpa) + Fr)  # mpa x mpa

            de = Dr @ Er  # mpa x mx
            T0 = Pr - 2 * Er.T @ de + de.T @ Fr @ de  # mx x mx
            Gr = pdinv_Array(np.eye(mx) + 1 / (n1 * gamma) * T0)  # mx x mx

            df = Dr @ Fr  # mpa x mpa
            twoi_df = 2 * np.eye(mpa) - df  # mpa x mpa

            kappa = (n1 * beta) / ((n1 * var_lambda) ** 4)

            Afrak = -1 / (n1 * var_lambda) ** 2 * (twoi_df @ Dr) - kappa * (twoi_df) @ de @ Gr @ de.T @ (twoi_df.T)
            Bfrak = kappa * (Gr @ de.T) @ (twoi_df.T)
            Cfrak = Bfrak.T
            Dfrak = -kappa * Gr

            # print(n1, Afrak.shape, Bfrak.shape, Cfrak.shape, Dfrak.shape, rff_Xc_tr.shape, rff_PAc_tr.shape)
            # Cr = 1/((n1*var_lambda)**2) * np.eye(n1) + (phi_PAc_tr@Afrak@phi_PAc_tr.T) + (phi_Xc_tr@Bfrak@phi_PAc_tr.T) + (phi_PAc_tr@Cfrak@phi_Xc_tr.T) + (phi_Xc_tr@Dfrak@phi_Xc_tr.T)

            i_df = np.eye(mpa) - df  # mpa x mpa
            Wr = 1 / ((
                              n1 * var_lambda) ** 2) * Pr + Er.T @ Afrak @ Er + Pr @ Dfrak @ Pr + Pr @ Bfrak @ Er + Er.T @ Cfrak @ Pr  # mx x mx

            # first part
            Qr = 1 / (n1 * gamma) * T0 + np.eye(mx)  # mx x mx
            Rr = np.linalg.cholesky(Qr)  # mx x mx
            first_part = np.sum(np.log(np.diag(Rr)))

            # second part
            etidf = Er.T @ i_df  # mx x mpa
            second_part = np.trace((np.eye(mx) - n1 * beta * Wr) @ (
                    Vr - 2 / (n1 * var_lambda) * etidf @ Ur + 1 / ((n1 * var_lambda) ** 2) * etidf @ Sr @ etidf.T))


            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * first_part + second_part / gamma) / 2

        CV = CV / k
    else:
        # induce matrix for X
        fX = feat_gen(X, MidWidth[Xi] * 2)
        phi_X = kernel_icd(fX, kerneld, m)

        mx = phi_X.shape[1]

        # centering
        phi_Xc = phi_X - np.mean(phi_X, 0)

        CV = 0
        for kk in range(k):
            if kk == 0:
                phi_Xc_te = phi_Xc[0:n0, :]
                phi_Xc_tr = phi_Xc[n0:T, :]
                nv = n0
            elif kk == k - 1:
                phi_Xc_te = phi_Xc[kk * n0: T, :]
                phi_Xc_tr = phi_Xc[0: kk * n0, :]
                nv = T - kk * n0
            elif 0 < kk < k - 1:
                idn = np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)])
                phi_Xc_te = phi_Xc[kk * n0: (kk + 1) * n0, :]
                phi_Xc_tr = phi_Xc[idn, :]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            Pr = phi_Xc_tr.T @ phi_Xc_tr
            Vr = phi_Xc_te.T @ phi_Xc_te

            Qr = 1 / (n1 * gamma) * Pr + np.eye(mx)
            Rr = np.linalg.cholesky(Qr)
            first_part = np.sum(np.log(np.diag(Rr)))

            Dr = pdinv_Array(np.eye(mx) + 1 / (n1 * gamma) * Pr)
            vp = Vr @ Pr
            second_part = np.trace(Vr - 1 / (n1 * gamma) * vp + (1 / (n1 * gamma) ** 2) * vp @ Dr @ Pr)


            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * first_part + second_part / gamma) / 2

        CV = CV / k


    score = CV  # negative cross-validated likelihood
    return score


def local_score_cv_multi_nym(
        Data: ndarray, Xi: int, PAi: List[int], parameters: Dict[str, Any]
) -> float:
    """
    Calculate the local score
    using negative k-fold cross-validated log likelihood as the score
    based on a regression model in RKHS

    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters:
                   kfold: k-fold cross validation
                   lambda: regularization parameter

    Returns
    -------
    score: local score
    """

    PAi = list(PAi)
    # print( parameters["dlabel"])
    T = Data.shape[0]
    indx = parameters["dlabel"][Xi]
    if len(indx) == 1:
        X = Data[:, indx].reshape(-1, 1)
    else:
        X = Data[:, indx]
    var_lambda = parameters["lambda"]  # regularization parameter
    k = parameters["kfold"]  # k-fold cross validation
    m = parameters["m"]
    MidWidth = parameters['MidWidth']
    # max_sample = parameters["max_sample"] # samples use to calculate the midwidth
    n0 = math.floor(T / k)
    gamma = 0.01
    beta = var_lambda ** 2 / gamma
    Thresh = 1e-5
    deltaa = 1e-8
    width_ratio = 3

    kerneld = DiagGaussianKernel()

    if len(PAi):
        # induce matrix for X
        fX = feat_gen(X, MidWidth[Xi] * width_ratio)
        phi_X = kernel_icd(fX, kerneld, m)
        mx = phi_X.shape[1]

        # induce matrix for PA

        legth_all_pai = 0
        # print(PAi)
        for i in PAi:
            # print(i)
            # ind_pai = parameters["dlabel"][PAi[i]]
            ind_pai = parameters["dlabel"][i]
            legth_all_pai += len(ind_pai)

        fPA = np.zeros((T, legth_all_pai))
        si = 0
        for i in range(len(PAi)):
            ind_pai = parameters["dlabel"][PAi[i]]
            ei = si + len(ind_pai)
            pa_par = Data[:, ind_pai]
            fPA_par = feat_gen(pa_par, MidWidth[PAi[i]] * width_ratio * np.sqrt(len(ind_pai)))
            fPA[:, si:ei] = fPA_par
            si = ei

        # induce matrix for PA
        phi_PA = kernel_icd(fPA, kerneld, m)
        mpa = phi_PA.shape[1]

        #         # calculate width (default 300)
        #         width_set_x = []
        #         width = kernel_midwidth_rbf(X[:max_sample,:])
        #         # print("midwidth for the variable X is ", width)
        #         width_set_x.append(width)

        #         width_set_pa = []
        #         for i in range(PA.shape[1]):
        #             width = kernel_midwidth_rbf(PA[:max_sample, i].reshape(-1,1))
        #             # print("midwidth for the variable Z", i, " is ", width)
        #             width_set_pa.append(width)

        # centering
        phi_Xc = phi_X - np.mean(phi_X, 0)
        phi_PAc = phi_PA - np.mean(phi_PA, 0)

        CV = 0
        for kk in range(k):
            if kk == 0:
                phi_Xc_te = phi_Xc[0:n0, :]
                phi_Xc_tr = phi_Xc[n0:T, :]
                phi_PAc_te = phi_PAc[0:n0, :]
                phi_PAc_tr = phi_PAc[n0:T, :]
                nv = n0  # sample size of validated data
            elif kk == k - 1:
                phi_Xc_te = phi_Xc[kk * n0: T, :]
                phi_Xc_tr = phi_Xc[0: kk * n0, :]
                phi_PAc_te = phi_PAc[kk * n0: T, :]
                phi_PAc_tr = phi_PAc[0: kk * n0, :]
                nv = T - kk * n0
            elif kk < k - 1 and kk > 0:
                idn = np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)])
                phi_Xc_te = phi_Xc[kk * n0: (kk + 1) * n0, :]
                phi_Xc_tr = phi_Xc[idn, :]
                phi_PAc_te = phi_PAc[kk * n0: (kk + 1) * n0, :]
                phi_PAc_tr = phi_PAc[idn, :]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            Pr = phi_Xc_tr.T @ phi_Xc_tr  # mx x mx
            Er = phi_PAc_tr.T @ phi_Xc_tr  # mpa x mx
            Fr = phi_PAc_tr.T @ phi_PAc_tr  # mpa x mpa
            Vr = phi_Xc_te.T @ phi_Xc_te  # mx x mx
            Ur = phi_PAc_te.T @ phi_Xc_te  # mpa x mx
            Sr = phi_PAc_te.T @ phi_PAc_te  # mpa x mpa

            Dr = pdinv_Array(n1 * var_lambda * np.eye(mpa) + Fr)  # mpa x mpa

            de = Dr @ Er  # mpa x mx
            T0 = Pr - 2 * Er.T @ de + de.T @ Fr @ de  # mx x mx
            Gr = pdinv_Array(np.eye(mx) + 1 / (n1 * gamma) * T0)  # mx x mx

            df = Dr @ Fr  # mpa x mpa
            twoi_df = 2 * np.eye(mpa) - df  # mpa x mpa

            kappa = (n1 * beta) / ((n1 * var_lambda) ** 4)

            Afrak = -1 / (n1 * var_lambda) ** 2 * (twoi_df @ Dr) - kappa * (twoi_df) @ de @ Gr @ de.T @ (twoi_df.T)
            Bfrak = kappa * (Gr @ de.T) @ (twoi_df.T)
            Cfrak = Bfrak.T
            Dfrak = -kappa * Gr

            # print(n1, Afrak.shape, Bfrak.shape, Cfrak.shape, Dfrak.shape, rff_Xc_tr.shape, rff_PAc_tr.shape)
            # Cr = 1/((n1*var_lambda)**2) * np.eye(n1) + (phi_PAc_tr@Afrak@phi_PAc_tr.T) + (phi_Xc_tr@Bfrak@phi_PAc_tr.T) + (phi_PAc_tr@Cfrak@phi_Xc_tr.T) + (phi_Xc_tr@Dfrak@phi_Xc_tr.T)

            i_df = np.eye(mpa) - df  # mpa x mpa
            Wr = 1 / ((
                              n1 * var_lambda) ** 2) * Pr + Er.T @ Afrak @ Er + Pr @ Dfrak @ Pr + Pr @ Bfrak @ Er + Er.T @ Cfrak @ Pr  # mx x mx

            # first part
            Qr = 1 / (n1 * gamma) * T0 + np.eye(mx)  # mx x mx
            Rr = np.linalg.cholesky(Qr)  # mx x mx
            first_part = np.sum(np.log(np.diag(Rr)))

            # second part
            etidf = Er.T @ i_df  # mx x mpa
            second_part = np.trace((np.eye(mx) - n1 * beta * Wr) @ (
                    Vr - 2 / (n1 * var_lambda) * etidf @ Ur + 1 / ((n1 * var_lambda) ** 2) * etidf @ Sr @ etidf.T))


            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * first_part + second_part / gamma) / 2

        CV = CV / k
    else:
        # induce matrix for X
        fX = feat_gen(X, MidWidth[Xi] * width_ratio)
        phi_X = kernel_icd(fX, kerneld, m)
        mx = phi_X.shape[1]

        # centering
        phi_Xc = phi_X - np.mean(phi_X, 0)

        CV = 0
        for kk in range(k):
            if kk == 0:
                phi_Xc_te = phi_Xc[0:n0, :]
                phi_Xc_tr = phi_Xc[n0:T, :]
                nv = n0
            elif kk == k - 1:
                phi_Xc_te = phi_Xc[kk * n0: T, :]
                phi_Xc_tr = phi_Xc[0: kk * n0, :]
                nv = T - kk * n0
            elif 0 < kk < k - 1:
                idn = np.concatenate([np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)])
                phi_Xc_te = phi_Xc[kk * n0: (kk + 1) * n0, :]
                phi_Xc_tr = phi_Xc[idn, :]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            Pr = phi_Xc_tr.T @ phi_Xc_tr
            Vr = phi_Xc_te.T @ phi_Xc_te

            Qr = 1 / (n1 * gamma) * Pr + np.eye(mx)
            Rr = np.linalg.cholesky(Qr)
            first_part = np.sum(np.log(np.diag(Rr)))

            Dr = pdinv_Array(np.eye(mx) + 1 / (n1 * gamma) * Pr)
            vp = Vr @ Pr
            second_part = np.trace(Vr - 1 / (n1 * gamma) * vp + (1 / (n1 * gamma) ** 2) * vp @ Dr @ Pr)


            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * first_part + second_part / gamma) / 2

        CV = CV / k

    score = CV  # negative cross-validated likelihood
    return score


class GaussianKernel:
    """
    A class to find gaussian kernel evaluations k(x, y) = exp (-||x - y||^2/2 sigma^2)
    """

    def __init__(self, sigma=1.0):
        """
        Initialise object with given value of sigma >= 0

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        """
        Find kernel evaluation between two matrices X1 and X2 whose rows are
        examples and have an identical number of columns.


        :param X1: First set of examples.
        :type X1: :class:`numpy.ndarray`

        :param X2: Second set of examples.
        :type X2: :class:`numpy.ndarray`
        """

        if X1.shape[1] != X2.shape[1]:
            raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        j1 = np.ones((X1.shape[0], 1))
        j2 = np.ones((X2.shape[0], 1))

        diagK1 = np.sum(X1 ** 2, 1)
        diagK2 = np.sum(X2 ** 2, 1)

        X1X2 = numpy.dot(X1, X2.T)

        Q = (2 * X1X2 - np.outer(diagK1, j2) - np.outer(j1, diagK2)) / (2 * self.sigma ** 2)

        return np.exp(Q)

    def __str__(self):
        return "GaussianKernel: sigma = " + str(self.sigma)


class DiagGaussianKernel:
    def __init__(self, sigma=1.0):
        """
        Initialise object with given value of sigma >= 0

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        # if X1.shape[1] != X2.shape[1]:
        # raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        K = numpy.exp(- numpy.sum((X1 - X2) ** 2, 1) / (2 * self.sigma ** 2))
        K = numpy.array(K, ndmin=2).T
        return K


def kernel_icd(X, kernel, m=100, precision=1e-6):
    """Approximates a kernel matrix using incomplete Cholesky decomposition (ICD).

    Input:	- X: data matrix in row format (each data point is a row)
                - kernel: the kernel function. It should calculate on the diagonal!
                - kpar: vector containing the kernel parameters.
                - m: maximal rank of solution
                - precision: accuracy parameter of the ICD method
    Output:	- G: "narrow tall" matrix of the decomposition K ~= GG'
                - subset: indices of data selected for low-rank approximation

    USAGE: G = km_kernel_icd(X,ktype,kpar,m,precision)

    Based on code from Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2010.

    The algorithm in this file is based on the following publication:
    Francis R. Bach, Michael I. Jordan. "Kernel Independent Component
    Analysis", Journal of Machine Learning Research, 3, 1-48, 2002.

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, version 3 (as included and available at
    http://www.gnu.org/licenses).
    """
    n = X.shape[0]

    perm = np.arange(n)  # permutation vector
    d = np.zeros((1, n))  # diagonal of the residual kernel matrix
    G = np.zeros((n, m))
    subset = np.zeros((1, m))

    for i in range(m):
        x = X[perm[i: n + 1], :]
        if i == 0:  # diagonal of kernel matrix
            d[:, i: n + 1] = kernel(x, x).T
        else:  # update the diagonal of the residual kernel matrix
            d[:, i: n + 1] = kernel(x, x).squeeze() - np.sum(G[i: n + 1, : i] ** 2, 1).T

        dtrace = np.sum(d[:, i:n + 1])

        if dtrace <= 0:
            print("Warning: negative diagonal entry: ", np.diag)

        if dtrace <= precision:
            G = G[:, :i]
            subset = subset[:i]
            break

        m2 = np.max(d[:, i: n + 1])  # find the new best element
        j = np.argmax(d[:, i: n + 1])
        # j = j + i - 1  #take into account the offset i
        j = j + i  # take into account the offset i
        m1 = np.sqrt(m2)
        subset[0, i] = j

        perm[[i, j]] = perm[[j, i]]  # permute elements i and j
        # permute rows i and j
        G[[i, j], :i] = G[[j, i], :i]
        G[i, i] = m1  # new diagonal element

        # Calculate the i-th column. May
        # introduce a slight numerical error compared to explicit calculation.

        G[i + 1: n + 1, i] = (kernel(X[perm[i], :], X[perm[i + 1:n + 1], :]).T -
                              np.dot(G[i + 1:n + 1, :i], G[i, :i].T)) / m1

    ind = np.argsort(perm)
    G = G[ind, :]
    return G
