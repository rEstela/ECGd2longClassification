import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf

print('...')
print('--> VERSOES')
print("Versao python:", sys.version)
print("Versao de Numpy:", np.__version__)


# -------------------------------------------------------------------------------------
def pixelflipping(img, coeff, segments):
    '''
    Gera um gráfico que demonstra se a remoção dos segmentos/superpixels destacados como mais importantes no método
    de interpretabilidade faz com que a performance do classificador caia significativamente. 
    Para isso, os segmentos/superpixels são removidos (set to zero) um a um recursivamente, ordenados do mais 
    relevante ao menos relevante, e é realizada a predição para cada imagem gerada.
    
    Args.
        img: (height x width x [1 or 3])
        coeff: (num_superpixels, ) coefficients vector
        segments: (height x width) superpixels
        
    Returns.
        Curvas individuais para a imagem analisada
        predictions: (num_superpixels, ) prediction vector
    
    '''
    
    print('#### RUNNING PIXELFLIPPING...')
    
    # -------------------------------------------------------------------------------------
    def remove_segment(img, remove, coeff, segments):
        '''
        Gera uma imagem com segmento removido (set to zero)

        Args.
            img: (height x width x [1 or 3])
            remove: (float) coeff value to be removed
            coeff: (num_superpixels, ) coefficients vector
            segments: (height x width) superpixels

        Returns.
            perturbed_image: (height x width x 3) 
            

        '''
        active_pixels = np.where(coeff == remove)[0]
        mask = np.ones(segments.shape)

        for active in active_pixels:
            mask[segments == active] = 0
            perturbed_image = copy.deepcopy(img)
            perturbed_image = perturbed_image * mask[:, :, np.newaxis]

        return perturbed_image
    # -------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------
    def plot_pred(predictions):
        '''
        Pixel-flipping curve for an individual image
        
        Args.
            predictions: (num_superpixels, ) prediction vector
            
            
        '''
        fig = plt.figure(figsize=(15,10))
        plt.plot(predictions, linewidth=2.0, c='0.85') # gray line
        plt.title("Pixel-flipping curve")
        plt.xlabel("Segments removed")
        plt.ylabel("Output score")
        plt.ylim(0, 1)
        plt.xlim(0, len(predictions))
        plt.grid()
    # -------------------------------------------------------------------------------------
    
    # sort coefficients
    top_coeff_idx = coeff.argsort()[::-1]
    top_coeff = coeff[top_coeff_idx]
    
    predictions = []
    
    # Prediction with all segments
    n_preds = model.predict(img)
    predictions.append(n_preds)

    # Removing 1st segment
    remove = top_coeff[0]
    x = remove_segment(img[0,...], remove, coeff, superpixels)
    n_preds = model.predict(x[np.newaxis,:,:,:])
    predictions.append(n_preds)

    # Removing segments recursively
    for i in range(len(coeff)-1):
        remove = top_coeff[i+1]
        x = remove_segment(x, remove, coeff, superpixels)
        n_preds = model.predict(x[np.newaxis,:,:,:]) 
        predictions.append(n_preds)
        
    predictions = np.array(predictions)
    predictions = predictions[:,0,0]
    plot_pred(predictions)    
    
    print('#### DONE.')

    return predictions
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def plot_pred_ind(predictions):
    '''
    Pixel-flipping curve for an individual image

    Args.
        predictions: (num_superpixels, ) prediction vector


    '''
    plt.plot(predictions, c='0.85', lw=1, alpha=0.5, label='Individual') # gray line
    plt.title("Pixel-flipping curve", fontsize=30)
    plt.xlabel("Number of flipped segments", fontsize=11)
    plt.ylabel("Output score", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, len(predictions)-1)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def plot_pred_mean(predictions):
    '''
    Pixel-flipping curve for mean predictions

    Args.
        predictions: (num_superpixels, n_img) 


    '''
    plt.plot(predictions, lw=2.0, ls='--', color='r', alpha=0.8, label='Mean')
    plt.title("Pixel-flipping curve", fontsize=30)
    plt.xlabel("Number of flipped segments", fontsize=11)
    plt.ylabel("Output score", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, len(predictions)-1)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
###################################
# Generic use of pixel-flipping   #
###################################

## ============================= ##
#  model => LOAD MODEL            #
#  x_img => LOAD IMAGE TO ANALYZE #
## ----------------------------- ##
#  Set Interpreter                #
# e.g.
superpixels, coeff, m_lime = LimeInterpreter(model, x_img, num_perturb, seg_method, grid_size)
## ----------------------------- ##
# Get Pixel-flipping
predictions = pixelflipping(x_img, coeff, superpixels)

## ============================= ##
#  Comparing XAI of a dataset     #
#  data => LOAD DATASET           #
p_flip = [] # get all predictions from pixel-flipping to create a mean curve
for idx in FAc1:
    x_img = data[idx]
    #  Set Interpreter                #
    # e.g.
    superpixels, coeff, m_lime = LimeInterpreter(model, x_img, num_perturb, seg_method, grid_size)
    ## ----------------------------- ##
    # Get Pixel-flipping
    predictions = pixelflipping(x_img, coeff, superpixels)
    p_flip.append(predictions)

# Plots #
fig = plt.figure()
# -> Individuals
for i in range(len(p_flip)):
    plot_pred_ind(p_flip[i])

# -> Mean
pflip = np.array(p_flip)
mflip = pflip.mean(0)
plot_pred_mean(mflip)