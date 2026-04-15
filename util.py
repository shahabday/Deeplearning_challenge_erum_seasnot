import matplotlib.pyplot as plt

labels_dict = {
    0: "Non-annotated",
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Floating Algae",
    4: "Natural Organic Material",
    5: "Ship",
    6: "Oil Spill",
    7: "Marine Water",
    8: "Sediment-Laden Water",
    9: "Foam",
    10: "Turbid water",
    11: "Shallow Water",
    12: "Waves & Wakes",
    13: "Oil Platform",
    14: "Jellyfish",
    15: "Sea snot"
}

labels_dict_short = {
    0: "NA",
    1: "MD",
    2: "DenS",
    3: "SpFA",
    4: "NatM",
    5: "Ship",
    6: "Oil",
    7: "MWater",
    8: "SLWater",
    9: "Foam",
    10: "TWater",
    11: "SWater",
    12: "Waves",
    13: "OilPlat",
    14: "Jellyfish",
    15: "Sea snot"
}

cmap_classes = plt.get_cmap("tab20", 16)
# generate a dict with color values for labels 0-15
labels_colors = {i: cmap_classes(i) for i in range(16)}
print(labels_colors)
