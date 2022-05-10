import time
import os
from os import listdir
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle

import tensorflow as tf
import h5py

import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

from matplotlib.colors import LinearSegmentedColormap
import shap # requires numpy 1.20.0


t0 = time.time()


print('...')
print('--> VERSOES')
print("Versao python:", sys.version)
print("Versao de tensorflow:", tf.__version__)
print("Versao de Numpy:", np.__version__)

# -------------------------------------------------------------------------------------
def pixelflipping(img, coeff, segments, ptype = 'data'):
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
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.xlim(0, len(predictions))
        plt.grid()
    # -------------------------------------------------------------------------------------
    
    # ------
    # Define if will use the data or random
    if ptype == 'data':
        # sort coefficients
        top_coeff_idx = coeff.argsort()[::-1]
        top_coeff = coeff[top_coeff_idx]

    elif  ptype == 'random':
        # random
        top_coeff = coeff
    # ------


    predictions = []
    
    # Prediction with all segments
    n_preds = model.predict(img/255) 
    predictions.append(n_preds)

    # Removing 1st segment
    remove = top_coeff[0]
    x = remove_segment(img[0,...], remove, coeff, segments)
    n_preds = model.predict(x[np.newaxis,:,:,:]/255)
    predictions.append(n_preds)

    # Removing segments recursively
    for i in range(len(coeff)-1):
        remove = top_coeff[i+1]
        x = remove_segment(x, remove, coeff, segments)
        n_preds = model.predict(x[np.newaxis,:,:,:]/255) 
        predictions.append(n_preds)
        
    predictions = np.array(predictions)
    predictions = predictions[:,0,0]
    #plot_pred(predictions)    
    
    print('#### DONE.')

    return predictions
# -------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------- #
def ShapInterpreter(model, x_img, nsamples, seg_method='grid', grid_size=[3,10]):
    '''
    Aplica o método Kernel SHAP (Local Interpreter) em uma imagem aplicada no modelo classificador.
    
    Args.
        model: modelo treinado
        x_img: (1, im_height, im_width, 1) imagem a ser analisada
        seg_method: str com o método de segmentação utilizado
        nsamples: número de perturbações geradas aleatoriamente
        grid_size: [h_ration, w_ration] int com quantidade de quadrados (h: height | w: width)
    
    Returns.
        Plots do resultado do shap
        
    '''
    print('#### RUNNING SHAP...')
    
    # Passo 1: Criar perturbações
    # -------------------------------------------------------------------------------------
    def to_rgb(img):
        '''
        Recebe imagem em grayscale (1 x height x width x 1) e copia para os outros canais, 
        retornando (1 x height x width x 3)

        '''
        x_rgb = np.zeros((img.shape[0],img.shape[1], img.shape[2], 3))

        for i in range(3):
            x_rgb[..., i] = img[..., 0]

        return x_rgb
    # -------------------------------------------------------------------------------------

    Xi = to_rgb(x_img) # converte para 3 canais
    Xi = Xi[0,...]

    # -------------------------------------------------------------------------------------
    # Segmentação

    # -------------------------------------------------------------------------------------
    def gridSegmentation(im, h_ratio, w_ratio):
        '''
        Define um Grid como Laberl array onde as regiões são marcadas por diferentes valores inteiros.

        Args.
            im: (im_height x im_width [, 3]) Grayscale or RGB image
            h_ration: [int] ration of squares in im_height
            w_ration: [int] ration of squares in im_width

        Returns.
            grid: (im_height x im_width [, 3]) Label array where regions are marked bu different integer values
        '''

        imgheight=im.shape[0]
        imgwidth=im.shape[1]

        i = 0
        M = imgheight//h_ratio
        N = imgwidth//w_ratio

        grid = np.ones((imgheight, imgwidth))

        for y in range(0,imgheight,M):
            for x in range(0, imgwidth, N):
                i = i+1
                grid[y:y+M, x:x+N] = i

        return grid
    # -------------------------------------------------------------------------------------
    
    #### ------------------------------ ####
    # Select segmentation method
    if seg_method == 'grid':
        superpixels = gridSegmentation(Xi, grid_size[0], grid_size[1]) # Segmentar em Grid
        
    elif seg_method == 'quickshift':
        superpixels = quickshift(Xi, kernel_size=10,max_dist=200, ratio=0.5)
        
    elif seg_method == 'slic':
        superpixels = slic(Xi, n_segments=20, compactness=10, sigma=1, start_label=1)
        
    elif seg_method == 'felzenszwalb':
        superpixels = felzenszwalb(Xi, scale=100, sigma=0.5, min_size=20)
        
    elif seg_method == 'watershed':
        gradient = sobel(rgb2gray(Xi))
        superpixels = watershed(gradient, markers=20, compactness=0.001)
        
    else:
        print('ERROR: Invalid segmentation method')
    
    # -------------------------------------------------------------------------------------
    # Criar perturbações aleatórias

    # Numero de superpixels
    num_superpixels = len(np.unique(superpixels))
    print('Number of superpixels: ', num_superpixels)
    
    # -------------------------------------------------------------------------------------
    # Passo 2: Definir função que gera uma máscara com os segmentos ON/OFF
    
    # -------------------------------------------------------------------------------------
    def mask_image(zs, segmentation, image, background=None):
        '''
        Gera uma máscara representando se uma região ficará ON ou OFF

        Args.
            zs: (1, num_superpixels) recebe do segundo argumento de KernelExplainer
            segmentation: (im_height x im_width [, 3]) Label array where regions are marked by different integer values
            image: (im_height x im_width x 3) RGB image
            background: [int] valor aplicado na máscara

        Returns.
            out: (num_superpixels x im_width x im_height x 3) mask image

        ''' 
        if background is None:
            background = image.mean((0,1))

        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))

        for i in range(zs.shape[0]):
            out[i,:,:,:] = image

            for j in range(zs.shape[1]):

                if zs[i,j] == 0:
                    out[i][segmentation == j,:] = background

         # treating output to fit the model input
        out = out[...,0]
        out = out[...,np.newaxis]

        return out
    # -------------------------------------------------------------------------------------

    # Passo 3: Definir função que retorna a predição
    # -------------------------------------------------------------------------------------
    def f(z):
        '''
        Retorna a predição da imagem de gerada na Func. mask_image, sendo que z é o segundo argumento de KernelExplainer
        '''    
        #return model.predict(mask_image(z, superpixels, Xi, 255)) # AdeleNet
        return model.predict(mask_image(z, superpixels, Xi, 255)/255) # ResNet
    # -------------------------------------------------------------------------------------

    # Passo 4: Computar SHAP
    # -------------------------------------------------------------------------------------
    # Kernel SHAP is a method that uses a special weighted linear regression to compute the importance of each feature.
    # The computed importance values are Shapley values from game theory and also coefficents from a local linear regression.
    explainer = shap.KernelExplainer(f, np.zeros((1,num_superpixels+1)))

    # Estima os valores shapley para o conjunto de nsamples geradas
    shap_values = explainer.shap_values(np.ones((1,num_superpixels+1)), nsamples=nsamples) # runs nsamples times
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    def fill_segmentation(values, segmentation):
        '''
        Preenche os segmentos com os valores associados a eles.

        '''
        out = np.zeros(segmentation.shape)

        for i in range(len(values)):
            out[segmentation == i] = values[i]

        return out
    # -------------------------------------------------------------------------------------

    m = fill_segmentation(shap_values[0][0], superpixels)
     
    return superpixels, shap_values[0][0], m

    
# -------------------------------------------------------------------------------------
def create_folders(path):
    '''
    Cria as pastas de destino para salvar os arquivos gerados
    

    Args:
       path: str with output path + archive name + extension (e.g. '../out_temp/' + img_name + '.png')
    '''
    
    if not os.path.exists(path):
        os.makedirs(path)
# -------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------- #
def img_gen(x_img, m, max_val ,idx, name, folder):
    '''
    Args.
        x_img: original image
        m: shap mask
        idx: idx of used image data
        name: string - FA | NORMAL
        
    Returns.
        SHAP result saved in destined folder
    '''
    # make a color map
    colors = []
    for l in np.linspace(0,1,100):
        colors.append((24/255, 196/255, 93/255,l))

    cm = LinearSegmentedColormap.from_list("shap", colors)
    
    # Plot
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title('Original\n imgID: ' + str(idx) + ' - class: ' + name)
    ax1.imshow(x_img[0], cmap='gray')
    ax1.axis('off')
    
    if max_val > 0.01:
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title('SHAP\n imgID: ' + str(idx) + ' - class: ' + name)
        ax2.imshow(x_img[0], cmap='gray')
        h = ax2.imshow(m, cmap = cm, vmin = -max_val, vmax = max_val)
        ax2.axis('off')
        h.set_clim(0,max_val)   
        plt.colorbar(mappable=h, label = "SHAP coeff", orientation = "horizontal", aspect = 60)
    else: # shapley values pequenos
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title('SHAP\n imgID: ' + str(idx) + ' - class: ' + name)
        ax2.imshow(x_img[0], cmap='gray')
        ax2.axis('off')
            
    #fig.savefig(folder + '/' + str(idx) + name + '_ecgD2_shap' + '.jpg')   
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
def plot_pred_ind(predictions, defcolor = 'c'):
    '''
    Pixel-flipping curve for an individual image

    Args.
        predictions: (num_superpixels, ) prediction vector


    '''
    if defcolor == 'c':
        #plt.plot(predictions, c='0.85', lw=1, alpha=0.5, label='Individual') # gray line
        plt.plot(predictions, color= 'r', lw=0.5, alpha=0.2, label='Individual') # blue thin line
    elif defcolor == 'b':
        plt.plot(predictions, color= defcolor, lw=0.5, alpha=0.2, label='Individual') # blue thin line
    #plt.title("Pixel-flipping curve", fontsize=30)
    plt.xlabel("Number of flipped segments", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, len(predictions)-1)
    #plt.grid()
    #plt.legend()
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def plot_pred_mean(predictions, defcolor = 'r'):
    '''
    Pixel-flipping curve for mean predictions

    Args.
        predictions: (num_superpixels, n_img) 


    '''
    plt.plot(predictions, lw=2.0, ls='-', color= defcolor, alpha=0.8, label='Mean')
    #plt.title("Pixel-flipping curve", fontsize=30)
    plt.xlabel("Number of flipped segments", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0, 1)
    plt.xlim(0, len(predictions)-1)
    #plt.grid()
    #plt.legend()
# -------------------------------------------------------------------------------------


# =====================================================================================
if __name__ == '__main__':
    # Caminho do diretório
    folder_dir = "/home/estela.ribeiro/erCodes/codes_D2_SHAP/ResNet6x30PixelFlipping/"

    # Cria o novo diretório
    create_folders(os.path.dirname(folder_dir))
    
    # Carregando modelo de classificação
    #model = tf.keras.models.load_model('modelD2adele.h5', compile=False)
    model = tf.keras.models.load_model('modelD2resnet.h5', compile=False)
    #model.summary()

    # Abrindo dataset de exemplo
    h5c1 = h5py.File('ecgD2longImgs.h5', 'r') # Open file
    h5c2 = h5py.File('ecgD2longImgsNEW.h5', 'r') # Open file
    datac1 = h5c1['2d'] # D2 images
    datac2 = h5c2['data'] # D2 images
    
    # Imagens correspondentes
    FAc1     = [8, 10, 17, 35, 37, 49, 60, 79]
    NORMALc1 = [3, 11, 25, 36, 44, 52, 68, 77]
    
    FAc2     = list(range(0,24))
    NORMALc2 = list(range(24,44))
    
    # set parameters
    grid_size   = [6, 30]
    nsample = 3000
    seg_method  = 'grid'
    
    p_flip = [] # get all predictions from pixel-flipping to create a mean curve
    rand_flip = [] # get predictions from pixel-flipping using random flip
    
    # ------------------------------------------------------------------------
    # Realizar método para cada imagem CONJ1
    for idx in FAc1:
        idx = idx
        name = 'FA'

        # load image
        x_img = datac1[idx]
        x_img = np.expand_dims(x_img, axis = 0) # input shape nedded for the model
        
        print('========================')
        print('## Image to be analyzed: ', idx)
        print('## Image prediction (NORMAL = 0 | FA = 1): ', model.predict(x_img/255))

        # -----------------
        # Aplica interpretador
        superpixels, shapvalue, m_shap = ShapInterpreter(model, x_img, nsample, seg_method, grid_size)
        # -----------------
        
        # Save
        #pd.DataFrame(superpixels).to_csv(folder_dir + '/' + str(idx) + name + '_ecgD2_SHAPsuperpixels' + '.csv')
        #pd.DataFrame(shapvalue).to_csv(folder_dir + '/' + str(idx) + name + 'ecgD2_SHAPvalue' +'.csv')
        # -----------------
        
        # Save
        #pd.DataFrame(superpixels).to_csv(folder_dir + '/' + str(idx) + name + '_ecgD2_SHAPsuperpixels' + '.csv')
        #pd.DataFrame(coeff).to_csv(folder_dir + '/' + str(idx) + name + 'ecgD2_SHAPcoeff' +'.csv')

        # image generator
        #img_gen(x_img, m_lime, np.max(coeff), idx, name, folder_dir)
        
        # -----------------
        # Aplica Pixel-flipping
        predictions = pixelflipping(x_img, shapvalue, superpixels, 'data')
        p_flip.append(predictions)
        
        # Aplica Pixel-flipping em Random
        l = list(range(len(shapvalue)))
        random.shuffle(l)
        ls = np.array(l)
        
        rand_predictions = pixelflipping(x_img, ls, superpixels, 'random')
        rand_flip.append(rand_predictions)
        # -----------------
    
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # Realizar método para cada imagem CONJ2
    for idx in FAc2:
        idx = idx
        name = 'FA'

        # load image
        x_img = datac2[idx]
        x_img = np.expand_dims(x_img, axis = 0) # input shape nedded for the model
        
        print('========================')
        print('## Image to be analyzed: ', idx)
        print('## Image prediction (NORMAL = 0 | FA = 1): ', model.predict(x_img/255))
        
        # -----------------
        # Aplica interpretador
        superpixels, shapvalue, m_shap = ShapInterpreter(model, x_img, nsample, seg_method, grid_size)
        # -----------------
        
        # Save
        #pd.DataFrame(superpixels).to_csv(folder_dir + '/' + str(idx) + name + '_ecgD2_SHAPsuperpixels' + '.csv')
        #pd.DataFrame(shapvalue).to_csv(folder_dir + '/' + str(idx) + name + 'ecgD2_SHAPvalue' +'.csv')
        # -----------------
        
        # Save
        #pd.DataFrame(superpixels).to_csv(folder_dir + '/' + str(idx) + name + '_ecgD2_SHAPsuperpixels' + '.csv')
        #pd.DataFrame(coeff).to_csv(folder_dir + '/' + str(idx) + name + 'ecgD2_SHAPcoeff' +'.csv')

        # image generator
        #img_gen(x_img, m_lime, np.max(coeff), idx, name, folder_dir)
        
        # -----------------
        # Aplica Pixel-flipping
        predictions = pixelflipping(x_img, shapvalue, superpixels, 'data')
        p_flip.append(predictions)
        
        # Aplica Pixel-flipping em Random
        l = list(range(len(shapvalue)))
        random.shuffle(l)
        ls = np.array(l)
        
        rand_predictions = pixelflipping(x_img, ls, superpixels, 'random')
        rand_flip.append(rand_predictions)
        # -----------------
    
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    
    # Save Pixel-flipping
    pd.DataFrame(p_flip).to_csv(folder_dir + '/' + name + '_ecgD2_SHAPpixelflipping' + '.csv')
    pd.DataFrame(rand_flip).to_csv(folder_dir + '/' + name + '_ecgD2_SHAPpixelflippingRAND' + '.csv')
    
     # Plots
    fig = plt.figure()
    # -> Individuals
    for i in range(len(p_flip)):
        plot_pred_ind(p_flip[i])

    # -> Mean
    pflip = np.array(p_flip)
    mflip = pflip.mean(0)
    plot_pred_mean(mflip, 'r')
    
    # -> Mean random
    rflip = np.array(rand_flip)
    mrflip = rflip.mean(0)
    plot_pred_mean(mrflip, 'b')
    
    # Save img
    fig.savefig(folder_dir + '/' + name + '_ecgD2_SHAPpixelflipping' + '.png')
# -------------------------------------------------------------------------------------

t1 = time.time()
exectime = t1-t0
print(exectime)

# Open a file and use dump()
with open(folder_dir + '/' + name + '_ecgD2_SHAPexecutiontime' + '.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(exectime, file)