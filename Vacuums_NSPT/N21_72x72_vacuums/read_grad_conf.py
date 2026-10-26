import numpy as np
import matplotlib.pyplot as plt

def read_cpn_conf(filename):
    with open(filename, 'rb') as f:
        # 1. Lettura dell'header in formato ASCII
        header_line = f.readline().decode('ascii').strip().split()
        update_index = int(header_line[0])
        N = int(header_line[1])
        L0 = int(header_line[2])
        L1 = int(header_line[3])
        # L'hash MD5 opzionale sarebbe in header_line[4], se presente
        
        volume = L0 * L1
        
        # 2. Definizione del tipo di dato per il blocco binario
        # Il codice C scrive la parte reale e immaginaria di ogni numero complesso come 'double' (8 byte).
        # Il byte order è forzato al Big-Endian tramite la funzione 'bytes_swap' in caso di macchine little-endian.
        dt = np.dtype([
            ('z', '>c16', (N,)),  # Vettore z di N numeri complessi a 128 bit (64 bit reale + 64 bit imm)
            ('U', '>c16', (2,))   # Variabili di link U nelle 2 direzioni spaziotemporali
        ])
        
        # 3. Lettura dei dati binari e reshape del reticolo
        # np.fromfile legge tutto in un colpo solo ottimizzando le prestazioni
        data = np.fromfile(f, dtype=dt, count=volume)
        
        # Riorganizza i dati in un reticolo bidimensionale L0 x L1
        lattice = data.reshape((L0, L1))
        
        return lattice, N, L0, L1, update_index

# Esempio di utilizzo:
filename = "outGrad_Q4.dat"
lattice, N, L0, L1, update_index = read_cpn_conf(filename)

# Esempio di visualizzazione grafica della norma (o componente 0) del campo z
z_field = lattice['z']  # array di shape (L0, L1, N)

# Creazione di una mappa di calore (heatmap) per il modulo quadrato della prima componente z_0
z0_squared_norm = np.abs(z_field[:, :, 0])**2

plt.imshow(z0_squared_norm, cmap='viridis', origin='lower')
plt.colorbar(label='|z_0|^2')
plt.title(f'Modulo Quadro z_0 - Update {update_index}')
plt.show()

# Estrazione delle variabili di link U
U_field = lattice['U']  # Questo array ha shape (L0, L1, 2)

# Calcoliamo la fase (angolo) dei numeri complessi in radianti [-pi, pi]
fase_U0 = np.angle(U_field[:, :, 0])  # Direzione mu = 0
fase_U1 = np.angle(U_field[:, :, 1])  # Direzione mu = 1

# Creiamo una figura con due grafici affiancati (uno per ogni direzione)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Grafico per mu = 0
# Usiamo 'twilight' perché è una colormap ciclica, perfetta per le fasi
im1 = ax1.imshow(fase_U0, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
ax1.set_title('Fase del Link U ($\mu=0$)')
fig.colorbar(im1, ax=ax1, label='Angolo (radianti)')

# Grafico per mu = 1
im2 = ax2.imshow(fase_U1, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
ax2.set_title('Fase del Link U ($\mu=1$)')
fig.colorbar(im2, ax=ax2, label='Angolo (radianti)')

fig.suptitle(f'Campo di Gauge (Variabili di Link U) - Update {update_index}')
plt.tight_layout()
plt.show()

# Estrazione dell'ultima componente (indice N-1)
z_last = z_field[:, :, N-1]

# Separazione di parte reale e immaginaria
z_last_real = np.real(z_last)
z_last_imag = np.imag(z_last)

# Creazione della figura con due grafici affiancati
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Grafico Parte Reale
# Usiamo un range di colori che metta in evidenza i valori positivi/negativi
im1 = ax1.imshow(z_last_real, cmap='viridis', origin='lower')
ax1.set_title(f'Parte Reale (z_{N-1})')
fig.colorbar(im1, ax=ax1, label='Re(z)')

# Grafico Parte Immaginaria
# Utile per verificare se il gauge fixing ha effettivamente azzerato questa parte
im2 = ax2.imshow(z_last_imag, cmap='plasma', origin='lower')
ax2.set_title(f'Parte Immaginaria (z_{N-1})')
fig.colorbar(im2, ax=ax2, label='Im(z)')

fig.suptitle(f'Analisi dell\'ultima componente z - Update {update_index}')
plt.tight_layout()
plt.show()