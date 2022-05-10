import os
from os import listdir

import sys
import h5py
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

from matplotlib.colors import LinearSegmentedColormap

print('...')
print('--> VERSOES')
print("Versao python:", sys.version)
print("Versao de tensorflow:", tf.__version__)
print("Versao de Numpy:", np.__version__)

# ---------------------------------------------------------------------------------- #
def LimeInterpreter(model, x_img, num_perturb, seg_method='grid', grid_size=[3,10]):
    '''
    Aplica o método LIME (Local Interpreter) em uma imagem aplicada no modelo classificador.
    
    Args.
        model: modelo treinado
        x_img: (1, im_height, im_width, 1) imagem a ser analisada
        seg_method: str com o método de segmentação utilizado
        num_perturb: número de perturbações geradas aleatoriamente
        grid_size: [h_ration, w_ration] int com quantidade de quadrados (h: height | w: width)
    
    Returns.
        Plots do resultado
        
    '''
    
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

    # Criar números aleatórios
    perturbations = np.random.binomial(1, 0.5, size = (num_perturb, num_superpixels))

    # -------------------------------------------------------------------------------------
    def perturb_image(img,perturbation,segments):
        '''
        Gera perturbações na imagem, baseado no vetor perturbations e nos superpixels (segments) definidos

        Args.
            img: (height x width x 3)
            perturbation: (num_perturb x num_superpixels)
            segments: (height x width) superpixels

        Returns.
            perturbed_image: (height x width x 3) com valores 0's e 1's correspondentes aos superpixels OFF e ON

        '''
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)

        for active in active_pixels:
            mask[segments == active] = 1
            perturbed_image = copy.deepcopy(img)
            perturbed_image = perturbed_image * mask[:, :, np.newaxis]

        return perturbed_image
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Passo 2: Usar o Modelo para predizer as classes das novas imagens geradas
    predictions = []
    for pert in perturbations:
        perturbed_img = perturb_image(Xi, pert, superpixels)
        x = perturbed_img[:, :, 0]
        x = x[np.newaxis, ..., np.newaxis]
        pred = model.predict(x/255)
        predictions.append(pred)    

    predictions = np.array(predictions)

    # -------------------------------------------------------------------------------------
    # Passo 3: Calcular distâncias entre imagem original e imagens perturbadas e computar os pesos 
    #(importâncias) de cada imagem perturbada
    original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled

    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()

    # -------------------------------------------------------------------------------------
    # Utilizar Função Kernel para computar os pesos
    kernel_width = 0.25

    weights = np.sqrt( np.exp ( - (distances ** 2) / kernel_width ** 2 ) ) #Kernel function

    # -------------------------------------------------------------------------------------
    # Passo 4: Utilizar `perturbations`, `predictions` e `weights` para gerar um modelo (linear) explicavel
    simpler_model = LinearRegression()
    simpler_model.fit(X = perturbations, y = predictions[:,:,0], sample_weight = weights)

    coeff = simpler_model.coef_[0]
    
    # -------------------------------------------------------------------------------------
    # Passo 5: Gerar a máscara com o resultado
    
    # -------------------------------------------------------------------------------------
    def fill_segmentation(values, segmentation):
        '''
        Preenche os segmentos com os valores associados a eles (máscara).
        
        Args.
            values: coeff in LIME | shap_value in SHAP
            segmentation: superpixels
            
        Returns.
            Mask

        '''
        out = np.zeros(segmentation.shape)

        for i in range(len(values)):
            out[segmentation == i] = values[i]

        return out
    # -------------------------------------------------------------------------------------

    # set fill segmentation
    m_lime = fill_segmentation(coeff, superpixels)
    
    return superpixels, coeff, m_lime

# ---------------------------------------------------------------------------------- #
def create_folders(path):
    '''
    Cria as pastas de destino para salvar os arquivos gerados
    

    Args:
       path: str with output path + archive name + extension (e.g. '../out_temp/' + img_name + '.png')
    '''
    
    if not os.path.exists(path):
        os.makedirs(path)

# ---------------------------------------------------------------------------------- #
def img_gen(x_img, m, max_val ,idx, name, folder):
    '''
    Args.
        x_img: original image
        m: LIME mask
        idx: idx of used image data
        name: string - FA | NORMAL
        
    Returns.
        Lime result saved in destined folder
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
        ax2.set_title('LIME\n imgID: ' + str(idx) + ' - class: ' + name)
        ax2.imshow(x_img[0], cmap='gray')
        h = ax2.imshow(m, cmap = cm, vmin = -max_val, vmax = max_val)
        ax2.axis('off')
        h.set_clim(0,max_val)   
        plt.colorbar(mappable=h, label = "LIME coeff", orientation = "horizontal", aspect = 60)
    else: # coeficientes muito pequenos
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title('LIME\n imgID: ' + str(idx) + ' - class: ' + name)
        ax2.imshow(x_img[0], cmap='gray')
        ax2.axis('off')
    
    fig.savefig(folder + '/' + str(idx) + name + '_ecgD2_lime' + '.jpg')   

# ---------------------------------------------------------------------------------- #
if __name__ == '__main__':
    # Caminho do diretório
    folder_dir = "/home/estela.ribeiro/erCodes/codes_D2_LIME/AdeleNet6x30NORMAL/"

    # Cria o novo diretório
    create_folders(os.path.dirname(folder_dir))
    
    # Carregando modelo de classificação
    model = tf.keras.models.load_model('modelD2adele.h5', compile=False) 
    #model = tf.keras.models.load_model('modelD2resnet.h5', compile=False) 
    #model.summary()

    # Abrindo dataset de exemplo
    h5 = h5py.File('ecgD2longImgs.h5', 'r') # Open file
    data = h5['2d'] # D2 images
    
    # Imagens correspondentes
    FA     = [8, 10, 17, 35, 37, 49, 60, 79]
    NORMAL = [3, 11, 25, 36, 44, 52, 68, 77]

    # ------------------------------------------------------------------------
    # Realizar método para cada imagem
    for idx in NORMAL:
        idx = idx
        name = 'NORMAL'

        # load image
        x_img = data[idx]
        x_img = np.expand_dims(x_img, axis = 0) # input shape nedded for the model

        print('## Image to be analyzed: ', idx)
        print('## Image prediction (NORMAL = 0 | FA = 1): ', model.predict(x_img/255))

        # set parameters
        grid_size = [6, 30]
        num_perturb = 6000
        seg_method = 'grid'

        # -----------------
        # Aplica interpretador
        superpixels, coeff, m_lime = LimeInterpreter(model, x_img, num_perturb, seg_method, grid_size)
        # -----------------
        
        # Save
        pd.DataFrame(superpixels).to_csv(folder_dir + '/' + str(idx) + name + '_ecgD2_LIMEsuperpixels' + '.csv')
        pd.DataFrame(coeff).to_csv(folder_dir + '/' + str(idx) + name + 'ecgD2_LIMEcoeff' +'.csv')

        # image generator
        img_gen(x_img, m_lime, np.max(coeff), idx, name, folder_dir)
# ---------------------------------------------------------------------------------- #