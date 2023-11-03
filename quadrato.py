import numpy as np
from numpy import fromfunction, roll
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import os

# Parametri del problema
maxIter = 200000  # Numero totale di iterazioni temporali
Re = 220.0  # Numero di Reynolds
nx, ny = 520, 180  # Dimensioni della griglia
ly = ny - 1.0
q = 9  # Numero di popolazioni in ciascuna direzione
cx, cy, r = nx // 4, ny // 2, ny // 9  # Coordinate del centro del quadrato
uLB = 0.04  # Velocità in unità di griglia
nulb = uLB * r / Re  # Viscosità in unità di griglia
omega = 1.0 / (3.0 * nulb + 0.5)  # Parametro di rilassamento

# Costanti del reticolo
c = np.array([(x, y) for x in [0, -1, 1] for y in [0, -1, 1]])  # Velocità del reticolo
t = 1.0 / 36.0 * np.ones(q)  # Pesi del reticolo
t[np.asarray([norm(ci) < 1.1 for ci in c])] = 1.0 / 9.0
t[0] = 4.0 / 9.0
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)]  # Indice delle condizioni di non scorrimento
i1 = np.arange(q)[np.asarray([ci[0] < 0 for ci in c])]  # Parete destra
i2 = np.arange(q)[np.asarray([ci[0] == 0 for ci in c])]  # Parete centrale verticale
i3 = np.arange(q)[np.asarray([ci[0] > 0 for ci in c])]  # Parete sinistra

# Funzioni di supporto
def sumpop(fin):
    return np.sum(fin, axis=0)

def equilibrium(rho, u):  # Funzione di equilibrio
    cu = 3.0 * np.dot(c, u.transpose(1, 0, 2))
    usqr = 3.0 / 2.0 * (u[0] ** 2 + u[1] ** 2)
    feq = np.zeros((q, nx, ny))
    for i in range(q):
        feq[i, :, :] = rho * t[i] * (1.0 + cu[i] + 0.5 * cu[i] ** 2 - usqr)
    return feq

# Inizializzazione
side_length = 2 * r  # Lato del quadrato
x0, y0 = cx - side_length / 2, cy - side_length / 2  # Coordinate dell'angolo in alto a sinistra del quadrato
obstacle = fromfunction(lambda x, y: (x >= x0) & (x < x0 + side_length) & (y >= y0) & (y < y0 + side_length), (nx, ny))

vel = fromfunction(lambda d, x, y: (1 - d) * uLB * (1.0 + 1e-4 * np.sin(y / ly * 2 * np.pi)), (2, nx, ny))
feq = equilibrium(1.0, vel)
fin = feq.copy()

# Percorsi delle immagini e della GIF
image_paths = []

# Loop principale
for time in range(maxIter):
    # Condizione di uscita
    fin[i1, -1, :] = fin[i1, -2, :]
    rho = sumpop(fin)
    u = np.dot(c.transpose(), fin.transpose((1, 0, 2))) / rho

    # Parete sinistra: calcolo della densità
    u[:, 0, :] = vel[:, 0, :]
    rho[0, :] = 1.0 / (1.0 - u[0, 0, :]) * (sumpop(fin[i2, 0, :]) + 2.0 * sumpop(fin[i1, 0, :]))

    # Condizione di Zou/He
    feq = equilibrium(rho, u)
    fin[i3, 0, :] = fin[i1, 0, :] + feq[i3, 0, :] - fin[i1, 0, :]

    # Collisione
    fout = fin - omega * (fin - feq)
    for i in range(q):
        fout[i, obstacle] = fin[noslip[i], obstacle]
    
    # Streaming
    for i in range(q):
        fin[i, :, :] = roll(roll(fout[i, :, :], c[i, 0], axis=0), c[i, 1], axis=1)

    # Visualizzazione e salvataggio immagini
    if time % 100 == 0:
        plt.clf()
        img_path = "vel." + str(time // 100).zfill(4) + ".png"
        plt.imshow(np.sqrt(u[0] ** 2 + u[1] ** 2).transpose(), cmap=cm.Reds)
        plt.savefig(img_path)
        image_paths.append(img_path)

# Creazione della GIF
with imageio.get_writer('flow_simulation.gif', mode='I') as writer:
    for img_path in image_paths:
        image = imageio.imread(img_path)
        writer.append_data(image)

# Rimozione delle immagini
for img_path in image_paths:
    os.remove(img_path)
