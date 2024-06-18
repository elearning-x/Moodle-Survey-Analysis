import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import csv
import io
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from svglib.svglib import svg2rlg


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)
                
        def draw(self, renderer):
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gridline in gridlines:
                    gridline.set_visible(False)
            super().draw(renderer)


    register_projection(RadarAxes)
    return theta


def MultiPie(mydf: pd, mytitle) -> plt:
    # défini le nb de colonnes
    if mydf.shape[1] == 1:
        ncols = 1
    else:
        ncols = 3

    # calcul du nb de ligne
    if mydf.shape[1] % ncols == 0:
        nrows = int(mydf.shape[1]/ncols)
    else:
        nrows = int(mydf.shape[1]/ncols)+1

    fig, axs = plt.subplots(figsize=(10, 10),
                            nrows=nrows,
                            ncols=ncols,
                            squeeze=False)
    fig.subplots_adjust(top=0.8)

    colors = ['#49DC3A',
              '#D0F741',
              '#5A45C3',
              '#3B7EBA',
              '#FFDA43',
              '#FFB143',
              '#FC424B',
              '#C634AF']
    for ax, (title, values) in zip(axs.flat, mydf.items()):

        ax.pie(values, labels=[k[0] for k, v in mydf.iterrows()],
               autopct=lambda p: f"{p:.2f}%\n({(p * sum(values)/100):,.0f})" if p > 0 else '',
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
               textprops={'color': "#8E8E90", 'weight': 'bold',
                          'fontsize': '14'},
               colors=colors)

        ax.set_title(mytitle, weight='bold',
                     size='medium',
                     position=(0.5, 1.1),
                     horizontalalignment='center',
                     verticalalignment='center')

    # récupère la légende et la place sur la figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,  bbox_to_anchor=(0, 0.8))

    # supprime les derniers graphes vides
    if mydf.shape[1] % ncols != 0:
        for n in range(ncols, mydf.shape[1] % ncols, -1):
            fig.delaxes(axs[nrows-1, n-1])

    fig.tight_layout(pad=2.0)
    # fig.show()
    return fig


def QPie(question, maxpage, course=''):
    title = f"{course}-{occurences[question].get('title')}"
    mydf = df[occurences[question]['column']].value_counts().sort_index(ascending=True).to_frame()
    myfig = MultiPie(mydf, f"{question}: {occurences[question].get('title')}")
    myfig.savefig(f"{remplacer_caracteres(title)}.svg")
    plt.close(myfig)
    occurences[question]['imgs'] = [f"{remplacer_caracteres(title)}.svg"]


def QRadar(question, course='') -> None:
    q = question
    # remplace les -999 par NaN
    # print(occurences[q])
    df[occurences[q]['column']] = df[occurences[q]['column']].replace(-999.0, np.nan)

    # vérfie que le tableau ne soit pas composé de valeurs vides
    if df[occurences[q]['column']].isna().all().all() == True:
        return
    
    categories = labels_radar(occurences[q]['column'])
    values = df[occurences[q]['column']].mean().tolist()
    title = f"{course}-{occurences[question].get('title')}"
        
    N = len(categories)

    theta = radar_factory(N, frame='polygon')
    fig1, axs = plt.subplots(figsize=(10, 10),
                            subplot_kw=dict(projection='radar'))
    fig1.subplots_adjust(wspace=0.25,
                        hspace=0.25,
                        top=0.85,
                        bottom=0.1)
    
    # Définir l'échelle maximale en fonction de la valeur maximale des moyennes
    max_val = np.ceil(max(values))

    axs.set_rgrids(np.arange(1, max_val+1, 1))
    axs.set_ylim(0, max_scale)  # reprend la valeur max de toutes les réponses

    # Tourner dans le sens des aiguilles
    axs.set_theta_direction(-1)

    # Déssine les polygones intérieurs
    for r in range(1, int(max_scale) + 1):
        polygon = RegularPolygon((0.5, 0.5), N,
                                 radius=r / max_val / 2,
                                 edgecolor="#D3D3D3",
                                 facecolor="none",
                                 transform=axs.transAxes)
        axs.add_patch(polygon)

    axs.plot(theta, values, color='#D0F741')
    axs.fill(theta, values, facecolor='#D0F741', alpha=0.25)
    axs.set_varlabels(categories)
    
    axs.set_title(title,
                weight='bold',
                size='medium',
                position=(0.5, 1.1),
                horizontalalignment='center',
                verticalalignment='center')
    
    #plt.yticks(np.arange(1, max_val+1, 1), color="grey", size=10)
    plt.yticks(np.arange(1, max_scale+1, 1), color="grey", size=10)
        
    fig1.tight_layout()
    # plt.show()
    fig1.savefig(f"{remplacer_caracteres(title)}.svg", format='svg')
    plt.close(fig1)
    occurences[question]['imgs'] = [f"{remplacer_caracteres(title)}.svg"]


def QHistoRadar(question, course='') -> None: 
    q=question
    # remplace les -999 par NaN
    df[occurences[q]['column']] = df[occurences[q]['column']].replace(-999.0, np.nan)

    title = f"{course}-{occurences[question].get('title')}"
    # calcul du nb de ligne
    nb_q = len(occurences[q]['column'])

    # vérfie que le tableau ne soit pas composé de valeurs vides
    if df[occurences[q]['column']].isna().all().all() == True:
        return
    
    if nb_q % 4 == 0:
        nb_lignes = nb_q // 4
    else:
        nb_lignes = (nb_q // 4) + 1        
    
    nb_cols = 4 if nb_q > 4 else nb_q  # Limiter le nombre de colonnes à 3 max

    # Ajuster dynamiquement la taille de la figure en fonction du nombre de lignes
    fig_width = 20
    fig_height = 4 * nb_lignes

    # Créer une grille de subplots avec n lignes et 3 colonnes max
    fig2, axes = plt.subplots(nrows=nb_lignes, ncols=nb_cols, figsize=(fig_width, fig_height))

    for idx, col in enumerate(df[occurences[q]['column']].columns):

        val_counts = df[col].value_counts(sort=False).sort_index()

        # Calculer les indices des subplots
        row = idx // nb_cols
        col_idx = idx % nb_cols
        # print(f'nb_lignes:{nb_lignes}, nb_cols:{nb_cols}, row:{row}, col_idx:{col_idx}')

        # Gérer le cas où axes est un tableau 1D ou 2D
        if nb_lignes == 1:
            ax = axes[col_idx]
        elif nb_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col_idx]

        # Créer l'histogramme à partir des données val_counts
        ax.bar(val_counts.index, val_counts.values, width=0.8, color='#A2CE00')

        # Ajouter des étiquettes et un titre
        ax.set_xlabel(f'Answers (average:{df[col].mean():,.3f})')
        ax.set_ylabel('Occurrences')
        ax.set_title(f'...{col[-30:]}')

        # Définir les ticks sur l'axe des x en tant qu'entiers croissants à partir de 1
        max_val = int(val_counts.index.max())
        ax.set_xticks(range(1, max_val + 1))

        # Définir les valeurs de l'axe des y en tant qu'entiers
        num_row = len(df[col])
        # Choisir le pas en fonction du nombre de réponses
        if num_row < 10:
            step = 1
        else:
            step = 10
        current_ylim = ax.get_ylim()
        ax.set_ylim(bottom=0, top=current_ylim[1])
        ax.set_yticks(range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1, step))

    # Supprimer les subplots vides si nécessaire
    if nb_q % nb_cols != 0:
        for idx in range(nb_q, nb_lignes * nb_cols):
            fig2.delaxes(axes.flat[idx])


    # Ajuster la mise en page pour éviter les chevauchements
    plt.tight_layout()

    # Afficher le graphique
    # plt.show()
    fig2.savefig(f"{remplacer_caracteres(title)}_hist.svg")
    plt.close(fig2)

    if 'imgs' not in occurences[question]:
        occurences[question]['imgs'] = [f"{remplacer_caracteres(title)}_hist.svg"]
    else:
        occurences[question]['imgs'].append(f"{remplacer_caracteres(title)}_hist.svg")

def labels_radar(columns: list) -> list:
    labels = []
    for str in columns:
        deb = str.find('>')
        label = multiligne(str[deb+1:], 30)
        labels.append(label)
    return labels


def multiligne(texte: str, size: int) -> str:
    mots = texte.split()
    resultat = ""
    ligne = ""
    for mot in mots:
        if len(ligne + mot) <= size:
            ligne += mot + " "
        else:
            resultat += ligne.strip() + "\n"
            ligne = mot + " "
    resultat += ligne.strip()
    return resultat


def remplacer_caracteres(chaine) -> str:
    # Créer une expression régulière qui correspond à espace, virgule ou deux points
    pattern = r'[ ,:;/]'
    # Remplacer tous les correspondances par '_'
    nouvelle_chaine = re.sub(pattern, '_', chaine)
    return nouvelle_chaine


def Plot_bar(question, maxpage, course=''):
    title = f"{course}-{occurences[question].get('title')}"
    # df[occurences[question]['column']].replace(-999.0, np.NaN, inplace=True)
    df.loc[:, occurences[question]['column']] = df.loc[:, occurences[question]['column']].replace(-999.0, np.nan)
    categories = labels_radar(occurences[question]['column'])
    values = df[occurences[question]['column']].mean().tolist()

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    bars = plt.bar(categories,
            values,
            color='#D0F741',
            width=0.4)

    plt.ylim((0, 5))
    plt.ylabel("Rating")
    plt.title(f"{title}")
    # plt.show()
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 height,
                f"{df[occurences[question]['column']].mean().iloc[idx]:,.3f}",
                 ha='center',
                 va='bottom')
    fig.savefig(f"{remplacer_caracteres(title)}.svg")
    plt.close(fig)
    occurences[question]['imgs'] = [f"{remplacer_caracteres(title)}.svg"]
    

def Df_Radar(table: pd.DataFrame, q:str ='', course:str =''):
    # vérfie que le tableau ne soit pas composé de valeurs vides
    if table.isna().all().all() == True:
        return
    
    # Préparer les données pour le graphique radar
    categories = labels_radar(list(table.columns))
    num_vars = len(categories)
    # Nombre total de groupes
    num_groups = len(table.index)
    # Nombre de lignes nécessaires
    if num_groups % 4 == 0:
        num_rows = num_groups // 4
    else:
        num_rows = (num_groups // 4) + 1        
    
    nb_cols = 4 if num_groups > 4 else num_groups  # Limiter le nombre de colonnes à 3 max
    
    # Configuration des subplots
    plt.figure(figsize=(20, nb_cols * num_rows))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    # Création d'un graphique radar pour chaque groupe
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'blue', 'green', 
        'red', 'cyan', 'magenta', 'yellow', 'black', 'darkorange', 
        'purple', 'brown', 'pink', 'gray', 'lightgreen', 'navy'
    ]
    for row in range(num_groups):
        # Indicateurs de catégorie
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        # La rotation des axes
        angles += angles[:1]
        
        # Valeurs du groupe + fermeture du cercle
        values = table.iloc[row].tolist()
        values += values[:1]
        
        ax = plt.subplot(num_rows, 4, row + 1, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, color='grey', size=8)
        ax.set_ylim(0, max_scale)  # force la limite à la valeur la plus haute des réponses
        ax.set_yticks(range(0, int(max_scale) + 1))

        # Tracer les lignes de chaque axe
        color = colors[row % len(colors)]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.25)

        title = table.index[row]
        plt.title(title, size=11, color=color, y=1.1)

    plt.savefig(f"{remplacer_caracteres(course)}{q}group_radar_charts.svg", format='svg')
    plt.close()

def Df_Pie(df:pd.DataFrame, q:str ='', course:str =''):
    # Détermination du nombre de lignes et de colonnes
    num_groups = len(df)
    num_cols = 4
    num_rows = (num_groups + num_cols - 1) // num_cols
    
    # Création des sous-graphiques
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
    
    # Si axes est 1D, le convertir en 2D pour simplifier l'itération
    if num_rows == 1:
        axes = [axes]
    
    colors = ['#49DC3A',
                  '#D0F741',
                  '#5A45C3',
                  '#3B7EBA',
                  '#FFDA43',
                  '#FFB143',
                  '#FC424B',
                  '#C634AF']
    
    # Fonction pour formater les labels des camemberts
    def autopct_format(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val:d})' if val != 0 else ''
        return my_autopct
    
    # Itération sur chaque groupe pour créer un graphique camembert
    for i, (group, values) in enumerate(df.iterrows()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row][col]

        ax.pie(values, autopct=autopct_format(values), startangle=90,
              wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, colors=colors)
        ax.set_title(group)
    
    # Supprimer les sous-graphiques inutilisés
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols][j % num_cols])
    
    # Créer une liste de labels pour la légende globale
    labels = df.columns.tolist()
    # Ajouter une légende globale
    fig.legend(labels, loc='upper left', ncol=len(labels))
    
    # Ajuster l'affichage
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{remplacer_caracteres(course)}{q}group_pie_charts.svg", format='svg')
    plt.close()


def create_pdf(pages=[], file_name='output.pdf'):
    """
    Create a PDF file with pages that contain SVG images resized to fit the page.

    Args:
        pages (list): A list of tuples, where each tuple contains the title of the page and a list of SVG file paths.
        file_name: PDF file name to create

    Returns:
        None
    """
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(file_name, pagesize=A4,
                            bottomMargin=0,
                            topMargin=.5*inch)
    width, height = A4
    elements = []

    for _type, page_title, data in pages:
        if (_type == 'img') & (data is not None):
            elements.append(Paragraph(page_title, styles["Heading1"]))
            elements.append(Spacer(1, 12))
            if len(data) > 1:
                available_height = (height - (2 * inch)) / len(data)
            else:
                available_height = height - (2 * inch)

            available_width = width - (2 * inch)
            # print(page_title)
            # print(f"width:{width} height:{height} - avail. W:{available_width} avail. H:{available_height}")
            
            for idx, svg_file in enumerate(data):
                drawing = svg2rlg(svg_file)                    
                scale_factor = min(available_width / drawing.width, available_height / drawing.height)
                # print(f"drawing.height:{drawing.height} Scale:{scale_factor} - new w:{drawing.width * scale_factor} new H:{drawing.height * scale_factor}")
                drawing.width *= scale_factor
                drawing.height *= scale_factor
                drawing.scale(scale_factor, scale_factor)
                elements.append(KeepTogether(drawing))
                if idx < len(svg_file):
                    elements.append(Spacer(1, 12))
            # Add a page break
            elements.append(PageBreak())
        elif _type == 'txt':
            elements.append(Paragraph(page_title, styles["Heading1"]))
            elements.append(Spacer(1, 12))
            # elements.append(Paragraph(data, styles["Normal"]))
            for line in data.splitlines(False):
                elements.append(Paragraph(line, styles["Normal"]))
            # Add a page break
            elements.append(PageBreak())

    doc.build(elements)


def generate_charts(change) -> str:
    """
    génère les graphique et retourne le nom de fichier en retour
    """
    global df
    global occurences

    home_files_path = os.path.abspath("files")
    # crée le dossier files s'il n'existe pas
    if not os.path.exists(home_files_path):
        os.makedirs(home_files_path)

    # on se place dans le dossier files
    os.chdir(home_files_path)

    # on fait le ménage
    existing_files = os.listdir(home_files_path)
    for file in existing_files:
        if file.endswith(".pdf"):
            os.remove( os.path.join( home_files_path, file ) )
        if file.endswith(".svg"):
            os.remove( os.path.join( home_files_path, file ) )
        if file.endswith(".png"):
            os.remove( os.path.join( home_files_path, file ) )
            
    #if type(change['new']) is dict: # 
    if isinstance(change['new'], dict):
        infos = change['new']
        infos = list(infos.values())[0]
        input_file_name = infos['metadata']['name']
    # elif type(change['new']) is tuple:
    elif isinstance(change['new'], tuple):
        infos = change['new'][0]
        input_file_name = infos['name']

    _temp = os.path.splitext(input_file_name)
    pdf_file_name = f"public.{_temp[0]}.pdf"

    content = infos['content']
    if isinstance(content, memoryview):
        content = bytes(content)
    content = io.StringIO(content.decode('utf-8'))

    # détermine la langue
    try:
        headers = pd.read_csv(content,
                              index_col=0,
                              nrows=0,
                              encoding='utf-8-sig').columns.tolist()
        if 'Course' in headers:
            course_col = 'Course'
            group_col = 'Group'
        else:
            course_col = 'Cours'
            group_col = 'Groupe'
    except:
        raise UserWarning("\nProblem reading the file. Check the name of the .CSV file\nEnter the correct file name in the previous step.\nProblème de lecture du fichier. Vérifier le nom du fichier .CSV\nSaisir le bon nom de fichier à l'étape précédente.")

    # cherche le séparateur
    content.seek(0)
    try:
        dialect = csv.Sniffer().sniff(content.read(1024), [',', ';'])
    except:
        raise UserWarning("Problème de lecture du fichier. Vérifier le nom du fichier .CSV\nSaisir le bon nom de fichier à l'étape précédente.")

    # charge le fichier
    content.seek(0)
    df = pd.read_csv(content,
                     encoding='utf-8-sig',
                     delimiter=dialect.delimiter,
                     on_bad_lines='skip')
    # print(df.head())

    if len(df.columns) <= 2:
        raise UserWarning(f'\nProblem reading the CSV file. Inconsistent number of columns ({df.columns}).\nProblème de lecture du fichier CSV. Nombre de colonnes incohérent ({df.columns})')

    # on supprime les colonnes d'identification (Institution, ID, nom, nom complet)
    df.drop(df.columns[[2,6,7,8]], axis=1, inplace=True)

    # on extrait les questions que l'on met dans un dictionnaire
    # questions = [col[:3] for col in df.columns if re.search(r"^Q\d{2}", col)]
    questions = [re.match(r"^(Q\d*)[_-]", elem).group(1) for elem in df.columns if re.match(r"^(Q\d*)[_-]", elem)]

    # on cherche le nombre de colonne associé à une question
    occurences = {}
    for q in questions:
        if q in occurences:
            occurences[q]['nb'] += 1
        else:
            occurences[q] = {}
            occurences[q]['nb'] = 1

    # on garde que les questions
    for col in df.columns:
        if re.match(r"^(Q\d*)[_-]", col):
            num_q = re.match(r"^(Q\d*)[_-]", col).group(1)
            indice_fin = col.find("-")
            if indice_fin == -1:
                indice_fin = len(col)
            occurences[num_q]['title'] = col[len(num_q)+1:indice_fin]
            if 'column' in occurences[num_q]:
                occurences[num_q]['column'].append(col)
            else:
                occurences[num_q]['column'] = [col]

    # on détermine le type de graphique
    for element in occurences.keys():
        if occurences[f"{element}"]['nb'] == 1:
            name = occurences[f"{element}"]['column'][0]

            if df.dtypes[name] == 'O':
                match = df[df[name].str.match(r"^[0-9]* :.*$") == True]
            else:
                # la colonne est numérique
                match = [1]

            if len(match) > 0:
                occurences[element]['type'] = 'pie'
            else:
                occurences[element]['type'] = 'comments'
        elif occurences[f"{element}"]['nb'] == 2:
            occurences[element]['type'] = 'bar'
        else:
            occurences[element]['type'] = 'radar'


    # Génération de la liste des cours triée alphabétiquement
    names = np.sort(df[course_col].unique())

    # -- Analyse des réponses pour l'echelle des radars --
    # Recherche de la position de la 1ere colonne de réponse
    Q01_index = df.columns.get_loc(occurences[list(occurences.keys())[0]]['column'][0])
    # Recherche la valeur max de toutes les valeurs max
    global max_scale
    max_scale = df.iloc[:,Q01_index:].max(axis=1, numeric_only=True).values.max()

    # Génère des DF pour chaque cours
    s_df = []
    for name in names:
        condition = df[course_col] == name
        filtered_rows = df.index[condition].tolist()
        s_df.append(df.loc[filtered_rows])

    # print(occurences)
    # Préparation des pages du PDF
    pages =[]

    # on parcours les sous DF
    for idx, df in enumerate(s_df):
        # Génération de DF  par groupes
        sg_s_df = []

        groups = np.sort(df[group_col].unique())
        for group in groups:
            condition = df[group_col] == group
            filtered_rows = df.index[condition].tolist()
            sg_s_df.append(df.loc[filtered_rows])

        # on crée le graphique pour chaque question
        for q in occurences:
            if occurences[q]['type'] == 'pie':
               # vérfie que la colonne ne soit pas vide
                if df[occurences[q]['column'][0]].isna().all().all() != True:
                    # on dessine un pie pour le cours entier
                    QPie(q, 9, course=names[idx])
                    # on enregistre les informations pour la page pdf
                    pages.append(('img', f"{names[idx]}, {q}: {occurences[q]['title']}", occurences[q].get('imgs')))
                # subplots par groupe
                result = df.groupby(group_col)[occurences[q]['column'][0]].value_counts().unstack(fill_value=0)
                if result.isna().all().all() != True:
                    Df_Pie(result, q=q, course=names[idx])
                    pages.append(('img', f"{names[idx]}, {q}: {occurences[q]['title']}", [f"{q}{names[idx]}group_pie_charts.svg"]))
            
            elif occurences[q]['type'] == 'bar':
                Plot_bar(q, 9, course=names[idx])
                # on enregistre les informations pour la page pdf
                pages.append(('img', f"{names[idx]}, {q}: {occurences[q]['title']}", occurences[q].get('imgs')))


            elif occurences[q]['type'] == 'radar':
                # vérfie que le tableau ne soit pas composé de valeurs vides
                if df[occurences[q]['column']].isna().all().all() != True:
                    #Crée le radar
                    QRadar(question=q, course=names[idx])
                    # Ajout de l'histogramme des réponses
                    QHistoRadar(question=q, course=names[idx])
                    # on enregistre les informations pour la page pdf
                    pages.append(('img', f"{names[idx]}, {q}: {occurences[q]['title']}", occurences[q].get('imgs')))

                # Sub plot par groupe
                table = pd.pivot_table(df, values=occurences[q]['column'], index=group_col)
                if table.isna().all().all() != True:
                    Df_Radar(table=table, q=q, course=names[idx])
                    pages.append(('img', f"{names[idx]} (groups), {q}: {occurences[q]['title']}", [f"{q}{names[idx]}group_radar_charts.svg"]))

            elif occurences[q].get('type') == 'comments':
                # on ajoute les commentaires pour le PDF
                comments = df[occurences[q]['column']].dropna()
                paragraph = ''
                for comment in comments.values:
                    if len(comment[0].strip()) > 0:
                        paragraph += f"{comment[0].strip()}\n"
                
                # on enregistre les informations pour la page pdf 
                if len(paragraph.split('\n')) >=2:
                    pages.append(('txt', f"{names[idx]}, {q}: {occurences[q]['title']}", paragraph))


    # Création du PDF
    create_pdf(pages=pages, file_name=pdf_file_name)

    return pdf_file_name

