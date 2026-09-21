import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# ==============================================================================
# ANEXO 0: CARREGAMENTO E CRIAÇÃO DAS IMAGENS COM RUÍDO
# ==============================================================================
def gerar_ruido_sal_pimenta(img, probabilidade=0.05):
    """Adiciona pontos brancos e pretos aleatórios (Salt & Pepper)."""
    ruidosa = np.copy(img)
    # Salt (Sal)
    pixels_sal = np.random.rand(*img.shape) < (probabilidade / 2)
    ruidosa[pixels_sal] = 255
    # Pepper (Pimenta)
    pixels_pimenta = np.random.rand(*img.shape) < (probabilidade / 2)
    ruidosa[pixels_pimenta] = 0
    return ruidosa

def gerar_ruido_gaussiano(img, media=0, desvio=20):
    """Adiciona um chuvisco granulado (Ruído Gaussiano/Eletrônico)."""
    gauss = np.random.normal(media, desvio, img.shape).astype(np.float32)
    img_ruidosa = cv2.add(img.astype(np.float32), gauss)
    return np.clip(img_ruidosa, 0, 255).astype(np.uint8)

print("Escolha uma imagem para começar a aula prática (mulher.tiff ou casa.tiff):")
uploaded = files.upload()

if uploaded:
    file_name = list(uploaded.keys())[0]
    img_original = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Gerando os ruídos para os experimentos
    img_sp = gerar_ruido_sal_pimenta(img_original)
    img_gauss = gerar_ruido_gaussiano(img_original)

    # Exibição Tripla
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title("1. Imagem Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_sp, cmap='gray')
    plt.title("2. Ruído Salt & Pepper\n(Desafio para Mediana)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_gauss, cmap='gray')
    plt.title("3. Ruído Gaussiano\n(Desafio para Gauss/Média)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # EXERCÍCIO 1: TAMANHO DO KERNEL (FILTRO DE MÉDIA)
    # ==========================================================================
    print("\n--- Executando Exercício 1: Tamanho do Kernel ---")
    
    # Kernel 3x3
    t0 = time.time()
    img_3x3 = cv2.blur(img_original, (3, 3))
    t_3x3 = (time.time() - t0) * 1000

    # Kernel 11x11
    t0 = time.time()
    img_11x11 = cv2.blur(img_original, (11, 11))
    t_11x11 = (time.time() - t0) * 1000

    # Kernel 31x31
    t0 = time.time()
    img_31x31 = cv2.blur(img_original, (31, 31))
    t_31x31 = (time.time() - t0) * 1000

    print(f"Tempo de execução - Kernel 3x3:   {t_3x3:.3f} ms")
    print(f"Tempo de execução - Kernel 11x11: {t_11x11:.3f} ms")
    print(f"Tempo de execução - Kernel 31x31: {t_31x31:.3f} ms")

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title("1. Original")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(img_3x3, cmap='gray')
    plt.title("2. Kernel 3x3\n(Vizinhança Pequena)")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(img_11x11, cmap='gray')
    plt.title("3. Kernel 11x11\n(Vizinhança Média)")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_31x31, cmap='gray')
    plt.title("4. Kernel 31x31\n(Vizinhança Grande)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # EXERCÍCIO 2: QUALIDADE DO BORRÃO - MÉDIA VS. GAUSSIANO
    # ==========================================================================
    print("\n--- Executando Exercício 2: Média vs. Gaussiano ---")
    k_size = (15, 15)
    
    # Filtro de Média (Box Filter)
    res_media = cv2.blur(img_original, k_size)
    
    # Filtro Gaussiano
    res_gauss = cv2.GaussianBlur(img_original, k_size, 0)

    plt.figure(figsize=(18, 9))
    plt.subplot(1, 3, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title("1. Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res_media, cmap='gray')
    plt.title(f"2. Filtro de Média\n(Janela {k_size[0]}x{k_size[1]})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res_gauss, cmap='gray')
    plt.title(f"3. Filtro Gaussiano\n(Janela {k_size[0]}x{k_size[1]})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # EXERCÍCIO 3: LIMPEZA DE RUÍDO
    # ==========================================================================
    print("\n--- Executando Exercício 3: Limpeza de Ruído ---")
    
    # Teste para Salt & Pepper (Mediana)
    restauro_sp = cv2.medianBlur(img_sp, 5)
    # Teste para Salt & Pepper com Gaussiano (demonstração de falha)
    sp_com_gauss = cv2.GaussianBlur(img_sp, (5, 5), 0)

    # Teste para Ruído Gaussiano (Gaussiano)
    restauro_gauss = cv2.GaussianBlur(img_gauss, (5, 5), 0)
    # Teste para Ruído Gaussiano com Mediana (demonstração cruzada)
    gauss_com_mediana = cv2.medianBlur(img_gauss, 5)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img_sp, cmap='gray')
    plt.title("A: Ruído Salt & Pepper (Original)")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(restauro_sp, cmap='gray')
    plt.title("Resultado: Filtro de Mediana (Correto)")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_gauss, cmap='gray')
    plt.title("B: Ruído Gaussiano (Chuvisco Original)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(restauro_gauss, cmap='gray')
    plt.title("Resultado: Filtro Gaussiano (Correto)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # EXERCÍCIO 4: INVESTIGANDO AS BORDAS (PADDING)
    # ==========================================================================
    print("\n--- Executando Exercício 4: Padding e Bordas ---")
    k_grande = (21, 21)

    # BORDER_REFLECT
    res_refletido = cv2.blur(img_original, k_grande, borderType=cv2.BORDER_REFLECT)
    
    # BORDER_CONSTANT (moldura preta/zero por padrão)
    res_constante = cv2.blur(img_original, k_grande, borderType=cv2.BORDER_CONSTANT)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(res_refletido, cmap='gray')
    plt.title("Reflect (Sem moldura artificial)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(res_constante, cmap='gray')
    plt.title("Constant (Moldura preta invadindo)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

else:
    print("Nenhum arquivo foi selecionado.")


'''
================================================================================
          RESPOSTAS TEÓRICAS E CONCEITUAIS DO ROTEIRO
================================================================================


--------------------------------------------------------------------------------
EXERCÍCIO 1: TAMANHO DO KERNEL
--------------------------------------------------------------------------------
1. O que acontece com os contornos dos objetos conforme o kernel aumenta?
   R: Conforme o tamanho do kernel aumenta, os contornos perdem nitidez e ficam 
   cada vez mais desfocados e suavizados. Isso ocorre porque o filtro de média 
   substitui cada pixel pelo valor médio do seu conjunto de vizinhos. Quanto maior 
   a vizinhança (de 3x3 para 31x31), maior é o espalhamento das transições abruptas 
   de intensidade, atenuando as altas frequências (bordas) e borrando a imagem.

2. Compare o tempo de resposta: kernels maiores demoram visivelmente mais?
   R: Em termos de complexidade de algoritmo pura, sim, kernels maiores realizam 
   mais operações por pixel (área $N \times N$). No entanto, as bibliotecas modernas 
   como a OpenCV utilizam otimizações de imagens integrais (Box Filter via Integral 
   Images), o que faz com que o tempo de execução permaneça extremamente baixo e 
   quase constante, sem aumentos significativos no tempo percebido.

3. Existe algum detalhe da imagem que desapareceu completamente com o kernel 31x31?
   R: Sim. Detalhes finos de alta frequência espacial desparecem por completo. 
   Na imagem "mulher.tiff", os fios de cabelo finos, o brilho dos olhos e as texturas 
   do chapéu somem. Na imagem "casa.tiff", os contornos das janelas distantes, 
   linhas de telhas e folhagens finas são eliminados, transformando-se em regiões 
   homogêneas e borradas.


--------------------------------------------------------------------------------
EXERCÍCIO 2: QUALIDADE DO BORRÃO - MÉDIA VS. GAUSSIANO
--------------------------------------------------------------------------------
1. Dê um zoom em uma área de borda (ex: o contorno de um prédio ou rosto). Qual dos dois 
   filtros parece mais "natural" e qual parece deixar rastros "quadrados" na imagem?
   R: O Filtro Gaussiano produz um desfoque visivelmente mais "natural" e suave. 
   O Filtro de Média deixa marcas e artefatos em forma de blocos "quadrados" 
   (box artifacts). Isso ocorre porque o Filtro de Média atribui peso igual (distribuição 
   uniforme) a todos os pixels da janela de convolução, enquanto o Filtro Gaussiano 
   atribui pesos ponderados em formato de sino, dando maior importância ao pixel 
   central e diminuindo suavemente a influência dos pixels conforme se afastam do centro.


--------------------------------------------------------------------------------
EXERCÍCIO 3: LIMPEZA DE RUÍDO
--------------------------------------------------------------------------------
1. Por que o Filtro de Mediana é o único que consegue remover os pontos brancos 
   e pretos sem borrar a imagem inteira?
   R: O ruído Salt & Pepper consiste em valores extremos de intensidade (0 ou 255). 
   O Filtro de Mediana é um filtro não linear estatístico de ordem: ele ordena os 
   pixels da vizinhança e escolhe o valor central (mediana). Como os valores 0 e 255 
   ficam nas extremidades da lista ordenada, eles são completamente descartados na 
   escolha do valor mediano, eliminando o ruído sem realizar o cálculo de médias que 
   borraria as bordas vizinhas.

2. O que aconteceu quando você tentou usar o Gaussiano no ruído Salt & Pepper?
   R: O Filtro Gaussiano falhou na remoção do ruído Salt & Pepper. Por ser um filtro 
   linear baseado na média ponderada, os valores extremos (0 e 255) entram no cálculo 
   aritmético, contaminando os pixels vizinhos. Em vez de apagar os pontos, o Gaussiano 
   apenas espalha o ruído, transformando os pontos isolados em manchas borradas e acinzentadas.


--------------------------------------------------------------------------------
EXERCÍCIO 4: INVESTIGANDO AS BORDAS (PADDING)
--------------------------------------------------------------------------------
1. O que apareceu nas extremidades da imagem?
   R: Ao utilizar `borderType=cv2.BORDER_CONSTANT`, surge uma moldura preta/escura 
   que invade as extremidades da imagem. Isso ocorre porque o algoritmo assume que 
   todos os pixels fora da imagem possuem valor igual a zero (preto). Durante a 
   convolução nas bordas, os zeros da moldura externa entram na soma da média, 
   escurecendo gradualmente as margens da imagem.

2. Por que em sistemas de reconhecimento facial ou leitura de placas, usar 
   BORDER_CONSTANT (moldura preta) pode ser um problema para o algoritmo?
   R: Porque o aparecimento repentino de uma moldura preta cria uma transição abrupta de 
   alta intensidade (uma borda falsa e muito forte) nos limites da imagem. Algoritmos 
   de detecção de características e extratores de bordas (como Canny, Sobel ou Haar 
   Cascades) podem identificar erroneamente essa margem preta artificial como uma borda 
   real do objeto ou da placa, gerando falsos positivos e interferindo no desempenho 
   do sistema de visão computacional.
================================================================================
'''