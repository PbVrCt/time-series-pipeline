import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.cm as cm
from sklearn import decomposition, pipeline, preprocessing, cluster


def pca(
    df,
    variance,
    transpose=False,
    pcs_vs=True,
    expl_var=True,
    loadings=False,
    row_referents=[],
    loading_referents=[],
):
    """
    Sklearn's pca + option to transpose the df + plots + plot options
          'transpose' : Boolean. Wether or not to transpose the df before performing pca
          'pcs_vs', 'expl_var', 'loadings', 'loadings_vs' : Booleans. Wether or not to show different plots
          'row_referents' : List of strings. Indexes of rows whose points are to be ploted in bigger size and/or different color as to set them apart in the figures.
                            If 'transpose = True' then it would be the indexes of columns before transposing the dataframe.
          'loading_referents' : List of strings. Names of original columns (before being reduced by pca) whose points/bars are to be ploted/barploted
                                in bigger size and/or different color, in the plots/barplots that plot principal component loadings.
                                If 'transpose = True' then it would be the indexes of rows before transposing the dataframe.
    """
    # Pca
    pca = decomposition.PCA(svd_solver="full", n_components=variance)
    scaler = preprocessing.StandardScaler()
    pipeline1 = pipeline.make_pipeline(scaler, pca)
    if transpose == True:
        df = df.T
    pc_data = pipeline1.fit_transform(df)
    loadings_data = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=["PC" + str(i) for i in range(1, len(pca.components_) + 1)],
    )
    # Plots
    col_names = df.columns.get_level_values(
        0
    )  # get_level_values(0) so it works on multiindex dataframes
    if pcs_vs == True:
        plot_pcs_vs(
            pc_data, variance, df.index, referents=row_referents, cl_labels=None
        )
    if expl_var == True:
        plot_explained_variance(pca)
    if loadings == True:
        plot_loadings(loadings_data, col_names, loading_referents)
    plt.show()
    return pc_data, loadings_data, pca


def pca_kmeans(
    df,
    variance,
    n_cl,
    max_cl,
    seed,
    transpose=False,
    pcs_vs=False,
    expl_var=False,
    row_referents=[],
):
    """
    'Pca' and 'plot_pcs_vs' same as above but now also doing kmeans clustering. For each point,
    a cluster label is given to 'plot_pcs_vs' so that points from different clusters are plotted in different colors.
          'n_cl' : Int. Number of clusters to make
          'max_cl' : Int. Maximum number of clusters to try inside 'calculate_WSS', a function that repeats the kmeans clustering with
                     different numbers of clusters, calculates wcss for each, and then plots: Within-cluster-sum-of-squares vs number of clusters
          'seed' : Int. Seed for the initialization of the kmeans algorithm. In a deployment scenario it should be random and not give problems.
           Other arguments same as above
    """
    # Pca
    pca_ = decomposition.PCA(svd_solver="full", n_components=variance)
    scaler = preprocessing.StandardScaler()
    pipeline1 = pipeline.make_pipeline(scaler, pca_)
    if transpose == True:
        df = df.T
    pc_data = pipeline1.fit_transform(df)
    # Kmeans
    kmeans = cluster.KMeans(n_clusters=n_cl, max_iter=1000, random_state=seed)
    kmeans.fit(pc_data)
    cl_labels = kmeans.predict(pc_data)
    clusters = {
        i: df.index[np.where(cl_labels == i)[0]].tolist() for i in np.unique(cl_labels)
    }  # cluster members #cambié columns por index
    # Plots
    if pcs_vs == True:
        plot_pcs_vs(
            pc_data,
            variance,
            df.index,
            referents=row_referents,
            cl_labels=cl_labels,
            title="Kmeans clusters by Pca components. On the graph only 1st and 2nd pc.",
        )
        calculate_WSS(pc_data, max_cl)
    if expl_var == True:
        plot_explained_variance(pca_)
    plt.show()
    return clusters


def plot_pcs_vs(
    pc_data,
    variance,
    rows,
    referents=[],
    cl_labels=None,
    title="1st and 2nd principal components",
):
    """
    Plot 1st pca component vs 2nd pca component. If there is just one component it plots that one.
          'rows' : List of indexes for each row (of the df on which pca was made, which could be the original df transposed). They are plotted as an annotation on top of the row's point
          'cl_labels' : Labels for each point, to paint points from different clusters in a different color.
          'referents' List of strings. List of indexes of rows whose points are ploted and annotated in bigger size, and also in different color if no cluster labels were provided.
          'variance' and 'title' are text boxes on the matplotlib figure
    """
    fig = plt.figure(title, figsize=(20, 20))
    ax = fig.add_subplot(111)
    if cl_labels is None:
        colors = ["b" if x not in referents else "g" for x in rows]
    else:
        ncl = len(np.unique(cl_labels))
        plt.text(0.1, 0.9, str(ncl) + " clusters", transform=ax.transAxes)
        colors = map_number_to_color(cl_labels)
    plt.text(
        0.1,
        0.93,
        str(len(pc_data[0, :])) + " principal components",
        transform=ax.transAxes,
    )
    plt.title("Enough Pca components for +{} explained variance".format(variance))
    plt.xlabel("1st pc")
    # Plots the 1st and the 2nd pc components
    if len(pc_data[0, :]) > 1:
        plt.ylabel("2nd pc")
        for i, txt in enumerate(rows):
            if txt in referents:
                plt.annotate(
                    txt, (pc_data[i, 0], pc_data[i, 1] - 0.1), fontsize=30, c=colors[i]
                )
                plt.scatter(pc_data[i, 0], pc_data[i, 1], s=250, c=colors[i], alpha=0.5)
            else:
                plt.annotate(
                    txt, (pc_data[i, 0] + 0.15, pc_data[i, 1]), fontsize=12, alpha=0.5
                )
                plt.scatter(pc_data[i, 0], pc_data[i, 1], s=50, c=colors[i])
    # If there is just one pca component it plots that one
    elif len(pc_data[0, :]) == 1:
        plt.yticks([], [])
        for i, txt in enumerate(rows):
            if txt in referents:
                plt.annotate(
                    txt,
                    (pc_data[i, 0], 1),
                    fontsize=25,
                    c=colors[i],
                    ha="left",
                    rotation=-45,
                    rotation_mode="anchor",
                )
                plt.scatter(pc_data[i, 0], 1, s=250, c=colors[i], alpha=0.5)
            else:
                plt.annotate(
                    txt,
                    (pc_data[i, 0], 1),
                    fontsize=12,
                    alpha=0.5,
                    ha="left",
                    rotation=-45,
                    rotation_mode="anchor",
                )
                plt.scatter(pc_data[i, 0], 1, s=50, c=colors[i])


def plot_explained_variance(pca):
    """
    Plot fraction of total variance explained by each principal component
    """
    fig = plt.figure(
        "Explained variance by amount of principal components"
    )  # plt.axes( [.65, .6, .2, .2] )   # Swap these lines to plot in the main plot
    plt.title(
        "Explained variance amount by each principal component \n Enough PCs for +{:.2f} explaine variance".format(
            pca.explained_variance_ratio_.sum() // 0.01 / 100
        ),
        fontsize=10,
    )
    plt.xlabel("Principal Components")
    plt.ylabel("Explained variance")
    xaxis = list(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.xticks(xaxis, xaxis)
    plt.bar(xaxis, pca.explained_variance_ratio_, width=0.4)


def plot_loadings(loadings, og_columns, loading_referents=[]):
    """
    For each principal component do a subplot and inside barplot its loadings from each of the original columns
          og_columns : List of strings. The original columns of the dataframe in which pca was performed.
          'loading_referents' : List of strings. Original columns to barplot in a different color as to differentiate them.
    """
    fig = plt.figure("Loadings", figsize=(20, 20))
    fig.text(0.5, 0.08, "Original features", ha="center")
    plt.title("PC loadings", fontsize=10)
    xaxis = list(range(1, len(og_columns) + 1))
    x = len(loadings.columns)
    nrows, ncolumns = vector_length_to_tile_dims(x)
    for i, pc in enumerate(loadings.columns):
        axs = plt.subplot(nrows, ncolumns, i + 1)
        axs.set_ylabel("{} Loadings".format(loadings.columns[i]))
        for j, x in enumerate(loadings[pc]):
            if og_columns[j] == loading_referents:
                axs.bar(xaxis[j], loadings[pc][j], width=0.4, color="green")
            else:
                axs.bar(xaxis[j], loadings[pc][j], width=0.4, color="blue")
        plt.xticks(xaxis, [])  # og_columns instead of [] to have xlabels


def calculate_WSS(pc_data, kmax):
    """
    Does Kmeans clustering of the rows of data provided. Repeats the clustering from k= 1 to k= 'kmax', calculates
    the 'Within Clusters Sum of Squares' and then plots this: 'Within Clusters Sum of Squares' vs 'Number of clusters'
    """
    sse = []
    for k in range(1, kmax + 1):
        kmeansModel = cluster.KMeans(n_clusters=k, max_iter=1000)
        kmeansModel.fit(pc_data)
        sse.append(kmeansModel.inertia_)
    # Plot Within-Cluster sum of squares
    plt.gcf()
    plt.axes([0.65, 0.6, 0.2, 0.2])
    plt.title("Within Clusters Sum of Squares")
    plt.xlabel("Number of clusters")
    xaxis = list(range(1, kmax + 1))
    plt.xticks(xaxis, xaxis)
    plt.scatter(xaxis, sse, alpha=0.7)


def map_number_to_color(value, cmap_name="Accent"):
    """
    from https://stackoverflow.com/questions/15140072/how-to-map-number-to-color-using-matplotlibs-colormap
    Transforms list of numbers to hex codes for colors
    """
    norm = clrs.Normalize()
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm([abs(x) for x in value]))
    color = [
        clrs.rgb2hex(x[:3]) for x in rgb
    ]  # As per stack overflow commenter: [cmap] "will return rgba, we take only first 3 so we get rgb"
    return color


def vector_length_to_tile_dims(n_subplots):
    """
    from https://stackoverflow.com/questions/31575399/dynamically-add-subplots-in-matplotlib-with-more-than-one-column
    For n subplots, finds squarish grid dimensions to fit them all. i.e. For 5 subplots it returns 2*3 grid dimensions
    """
    n_cols = np.ceil(np.sqrt(n_subplots))
    n_rows = np.ceil(n_subplots / n_cols)
    grid_shape = int(n_rows), int(n_cols)
    return grid_shape
