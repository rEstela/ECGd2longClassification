# XAI_D2long

Este é um repositório desenvolvido em Python (3.7) com aplicações de técnicas de Explainable AI (XAI) nos modelos ResNet50 e AdeleNet.
Nele são utilizadas imagens de traçados de ECG da derivação D2 longa, adquiridas da base de dados do InCor.

Este projeto é desenvolvido em conjunto com o pesquisador Felipe Menguetti Dias, baseado na publicação DOI: 10.22489/CinC.2021.189.

O objetivo é aplicar os métodos para compreender como a rede está chegando em suas predições, ou seja, "para onde a rede está olhando" para definir as classes FA e NORMAL.

Os métodos aplicados são:

* Grad-Cam;
* LIME;
* Kernel SHAP.

Para a avaliação dos métodos de interpretabilidade, é utilizado o método:
    
    
* Pixel-Flipping. 

Os resultados estão sendo analisados e um artigo está sendo produzido para ser submetido a uma revista de relevância na área.