from geo_tools import *
from utils import *
import pandas as pd
import matplotlib as mpl


def rose_plot(ax, y, p):
    ax.bar(np.radians(y.theta), 0.98, width=np.pi / 4, bottom=y.idh, color=y.color)
    ax.grid(True, linestyle="--", linewidth=2 * 0.005, zorder=0, alpha=0.25)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
    ax.set_xticklabels(
        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        fontsize=2 * 4,
        fontweight="bold",
        position=(0.1, 0.1),
    )
    ax.set_yticks(np.arange(y.idh.max() + 1))
    ax.set_yticklabels(
        [
            "0",
            "",
            "",
            "900",
            "",
            "",
            "1800",
            "",
            "",
            "2700",
            "",
            "",
            "3600",
            "",
            "",
            "",
        ],
        fontsize=5,
        color="red",
        fontweight="bold",
    )
    ax.set_rlabel_position(180)
    ax.invert_yaxis()
    ax.set_title(f"{p}°", fontsize=10, fontweight="bold")


def slope_full_rose(name, df_train, borne, cb):
    pente = [0, 20, 45]
    f, ax = plt.subplots(
        1,
        len(pente),
        subplot_kw=dict(projection="polar"),
        figsize=(2 * 8.5 / 2.54, 2 * 4 / 2.54),
    )
    viridis = mpl.colormaps["terrain_r"]
    norm = mpl.colors.LogNorm(vmin=borne[0], vmax=borne[1])
    for n, p in enumerate(pente):
        y = df_train[df_train.nabla == p].loc[:, ["alt", "idh", "theta", "s"]]
        y["color"] = y.s.apply(lambda x: mpl.colors.to_hex(viridis(norm(x))))
        rose_plot(ax[n], y, p)
    plt.tight_layout()
    if cb:
        cbar = f.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=viridis),
            ax=ax,
            orientation="horizontal",
            aspect=50,
            pad=0.1,
        )
        cbar.ax.tick_params(labelsize=2 * 5)
        cbar.set_label(label="Area km$^2$", fontsize=2 * 6)
    exist_create_folder("fig")
    plt.savefig(f"fig/{name}.pdf", backend="pgf")


def altitude_plage(plg):
    h = np.arange(0, 4800, 300)
    if plg[0] > h[-1]:
        altitude = h[-1]
    else:
        bmin = np.where(h > plg[0])[0][0]
        bmax = np.where(h > plg[1])[0][0]
        if (bmin == bmax) or (h[bmin] - plg[0] > plg[1] - h[bmin]):
            altitude = h[bmin - 1]
        else:
            altitude = h[bmin]
    return altitude


def theta_plage(theta):
    t0 = np.arange(22.5, 360 + 22.5, 45)
    t = np.arange(0, 360, 45)
    if theta > t0[-1] or theta < t0[0]:
        ct = t[0]
    else:
        ind = np.where(theta < t0)[0][0]
        ct = t[ind]
    return ct


def pente_plage(p):
    if p < 2:
        pente = 0
    elif p < 30:
        pente = 20
    else:
        pente = 45
    return pente


def combination():
    comb = []
    h = np.arange(0, 4800, 300)
    theta = np.arange(0, 360, 45)
    nabla = np.array([0, 20, 45])
    for i in h:
        for k in nabla:
            for j in theta:
                comb.append((i, k, j))
    return np.array(comb)


def count_comb(comb, hpo):
    gcou = []
    for i in comb:
        gcou.append(np.all(hpo == i, axis=1).sum())
    return np.array(gcou)


def get_hpo(chart):
    hpo = []
    img = chart.reshape(-1, chart.shape[-1])
    for i in range(img.shape[0]):
        high = (img[i, 3], img[i, 3])
        theta = img[i, 4]
        pente = img[i, 5]
        if pente != -999.0 and high[0] != -999.0:
            hpo.append((altitude_plage(high), pente_plage(pente), theta_plage(theta)))
    hpo = np.array(hpo)
    return hpo


def stat_crocus_massif(comb, file):
    chart = load_data(file)[0]
    hpo = get_hpo(chart)
    if hpo.shape[0] != 0:
        class_pixel = 10 * 10 * count_comb(comb, hpo) * 1e-6
        return class_pixel


def get_stats_massif(path, names_massif):
    comb = combination()
    class_massf = {}
    for n in names_massif:
        files = glob.glob(path + f"{n}*.tif")
        stats = []
        for f in tqdm(files):
            res = stat_crocus_massif(comb, f)
            if res is not None:
                stats.append(res)
        class_massf[n] = np.array(stats).sum(axis=0)
        class_massf["x"] = comb
    return class_massf


def extract_topo_dataset(fp, train, test):
    names_massif = train + test
    stats = get_stats_massif(fp, names_massif)
    q = np.sum([(stats[i]) for i in train], axis=0)
    df_train = np.concatenate([stats["x"], q.reshape(-1, 1)], axis=1)
    df_train = pd.DataFrame(df_train, columns=["alt", "nabla", "theta", "s"])

    q = np.sum([(stats[i]) for i in test], axis=0)
    df_test = np.concatenate([stats["x"], q.reshape(-1, 1)], axis=1)
    df_test = pd.DataFrame(df_test, columns=["alt", "nabla", "theta", "s"])

    df_train["idh"] = df_train.alt // 300
    df_test["idh"] = df_test.alt // 300
    return df_train, df_test
