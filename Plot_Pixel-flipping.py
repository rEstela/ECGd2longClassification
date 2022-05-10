import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------
def plot_pred_ind(predictions):
    '''
    Pixel-flipping curve for an individual image

    Args.
        predictions: (num_superpixels, ) prediction vector


    '''
    plt.plot(predictions, c='0.85', lw=1, alpha=0.5) # gray line
    #plt.title("Pixel-flipping curve", fontsize=30)
    plt.xlabel("Number of flipped segments", fontsize=11)
    plt.ylabel("Output score", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, len(predictions)-1)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def plot_pred_mean(predictions):
    '''
    Pixel-flipping curve for mean predictions

    Args.
        predictions: (num_superpixels, n_img) 


    '''
    plt.plot(predictions, lw=2.0, ls='-', color='r', alpha=0.8, label='Average')
    #plt.title("Pixel-flipping curve", fontsize=30)
    plt.xlabel("Number of flipped segments", fontsize=11)
    plt.ylabel("Output score", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, len(predictions)-1)
    plt.legend()
# ------------------------------------------------------------------------------

# Filename to save
savefilename = 'ResNet9x60_SHAPpixelflipping'

# Read file of XAI flipping
df = pd.read_csv ('PixelFlipping - RANDOM\ResNet9x60PixelFlipping\FA_ecgD2_SHAPpixelflipping.csv')
df = df.drop(['Unnamed: 0'], axis=1)
x = df.to_numpy()

# Read file of random flipping
dfrand = pd.read_csv ('PixelFlipping - RANDOM\ResNet9x60PixelFlipping\FA_ecgD2_SHAPpixelflippingRAND.csv')
dfrand = dfrand.drop(['Unnamed: 0'], axis=1)
xrand = dfrand.to_numpy()

# ---------------------------------------------------------------------------- #
####                                PLOTS                                   ####
# ---------------------------------------------------------------------------- #

# XAI flipping result
fig = plt.figure()
# -> Individuals
for i in range(len(x)):
    plot_pred_ind(x[i])

# -> Mean
x = np.array(x)
mx = x.mean(0)
plot_pred_mean(mx)

# Save img
fig.savefig(savefilename + '_explainableflip', format='eps', dpi=1000)
fig.savefig(savefilename + '_explainableflip' + '.png', dpi=1000)


# RANDOM flipping result
fig = plt.figure()
# -> Individuals
for i in range(len(xrand)):
    plot_pred_ind(xrand[i])

# -> Mean
xrand = np.array(xrand)
mxrand = xrand.mean(0)
plot_pred_mean(mxrand)
fig.savefig(savefilename + 'randomflip', format='eps', dpi=1000)
fig.savefig(savefilename + 'randomflip' + '.png', dpi=1000)

# ---------------------------------------------------------------------------- #
####         GENERATE A PLOT WITH XAI flipping and RANDOM flipping          ####
# ---------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------
def plot_pred_ind_rand(predictions, defcolor = 'c'):
    '''
    Pixel-flipping curve for an individual image

    Args.
        predictions: (num_superpixels, ) prediction vector


    '''
    if defcolor == 'c':
        #plt.plot(predictions, c='0.85', lw=1, alpha=0.5, label='Individual') # gray line
        plt.plot(predictions, color= 'r', lw=0.5, alpha=0.2) # blue thin line
    elif defcolor == 'b':
        plt.plot(predictions, color= defcolor, lw=0.5, alpha=0.2) # blue thin line
    #plt.title("Pixel-flipping curve", fontsize=30)
    plt.xlabel("Number of flipped segments", fontsize=11)
    plt.ylabel("Output score", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, len(predictions)-1)
    #plt.grid()
    #plt.legend()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
def plot_pred_mean_rand(predictions, defcolor = 'r', label = 'Average'):
    '''
    Pixel-flipping curve for mean predictions

    Args.
        predictions: (num_superpixels, n_img) 


    '''
    plt.plot(predictions, lw=2.0, ls='-', color= defcolor, alpha=0.8, label=label)
    #plt.title("Pixel-flipping curve", fontsize=30)
    plt.xlabel("Number of flipped segments", fontsize=11)
    plt.ylabel("Output score", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, len(predictions)-1)
    #plt.grid()
    plt.legend()
# ------------------------------------------------------------------------------

# ---------------------------------------------------------------------------- #
####                                PLOTS                                   ####
# ---------------------------------------------------------------------------- #

fig = plt.figure()
# XAI flipping result
# -> Individuals
for i in range(len(x)):
    plot_pred_ind_rand(x[i], 'c')

# -> Mean
plot_pred_mean_rand(mx, 'r', 'SHAP')

# RANDOM flipping result
# -> Indviduals random
for i in range(len(xrand)):
    plot_pred_ind_rand(xrand[i], 'b')

# -> Mean random
plot_pred_mean_rand(mxrand, 'b', 'Random')

fig.savefig(savefilename, format='eps', dpi=1000)
fig.savefig(savefilename + '.png', dpi=1000)