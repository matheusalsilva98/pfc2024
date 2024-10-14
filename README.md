# Geração de mosaicos de imagens CBERS-4A livres de nuvens utilizando aprendizado profundo
***
Projeto desenvolvido durante o Curso de Graduação em Engenharia Cartográfica pelo IME como Projeto de FInal de Curso (PFC).

## Resumo
No presente trabalho foi desenvolvida uma ferramenta capaz de gerar, de forma automatizada, mosaicos de ortoimagens provenientes do satélite CBERS-4A (Sensor WPM), tendo como objetivo a minimização da cobertura de nuvens, sendo empregada uma rede neural convolucional (CNN), chamada U-Net. Nesse sentido, foi construído um robusto conjunto de dados para servir de treinamento da rede em segmentar as nuvens e suas sombras. Tal ferramenta se propõe a exercer grande relevância na automatização das demandas existentes no que se refere à criação e edição de cartas ortoimagens nos cinco Centros de Geoinformação do Exército Brasileiro. 

Em suma, pode-se afirmar que, após a realização dos diversos treinamentos e testes, foi possível obter uma ferramenta com capacidade de reduzir consideravelmente as nuvens de oclusão do terreno em condições específicas, corroborando para a viabilidade da ferramenta com os devidos ajustes. Ademais, deve-se destacar que a ferramenta em questão possui meios para sua otimização, que foram abordados ao final dos resultados pelos autores.

## Objetivos do projeto
* **Objetivo principal**:
   * Elaborar uma ferramenta com capacidade de gerar mosaicos de imagens ópticas livres de nuvens e suas respectivas sombras, provenientes do sensor WPM do satélite CBERS-4A, a partir de imagens afetadas por tais fenômenos naturais.
* **Objetivos específicos**:
   *  Proporcionar a disponibilização de uma robusta fonte de dados com imagens e máscaras de segmentação, de modo a servir como fonte dos mais diversos algoritmos de classificação de imagens em pesquisas e estudos vindouras.
   * Demonstrar a utilização da ferramenta de geração sintética de nuvens e sombras __SatelliteCloudGenerator__ em imagens do sensor WPM do satélite CBERS-4A como fonte de dados para o treinamento de redes neurais convolucionais.

## Abordagem metodológica

Inicialmente, buscou-se construir o *Dataset* de treinamento, composto por imagens CBERS-4A/WPM com resolução espacial de 2 m e 6 bandas espectrais (R,G,B,NIR,NDVI,WI), sendo o NDVI e o Wi índices espectrais adicionados às imagens, e as respectivas máscaras de segmentação descrevendo as 4 classes (**_background_**, **nuvem densa**, **nuvem fina**, **sombra**). 

Logo abaixo segue um exemplo desse par imagem/máscara:

<img src="https://github.com/user-attachments/assets/87524304-2cc1-4335-b243-1c28eae95445" width="500px" height="250px"/>

A metodologia de treinamento e testes da U-Net consistiu, basicamente, nos fluxogramas descritos abaixo:

* Treinar a rede neural convoluciona U-Net:

<img src="https://github.com/user-attachments/assets/2d01f869-d2d8-4fcd-95ec-60823ae9c4bc" width="800px" height="600px"/>

* Realizar os testes com os pesos salvos dos treinamentos realizados:

<img src="https://github.com/user-attachments/assets/21650c50-2f52-4123-ac61-6f505f0a00ea" width="800px" height="600px"/>

## Resultados

Todas os resultado gerados, que se resumem nas máscaras de inferência das cenas de teste coletadas e nos mosaicos dos melhores resultados, podem ser encontrada no Google Drive no link abaixo:

Acesse o link do Google Drive [__aqui__](https://drive.google.com/drive/folders/12FPAsRHy8TSv26dNB28zOAmXbfW_RV-v?usp=sharing)

Segue abaixo um exemplo dos resultados obtidos:

<img src="https://github.com/user-attachments/assets/42c463f6-e442-4f31-afa7-b9c4beaf8ce3" width="100px" height="100px"/>
<img src="https://github.com/user-attachments/assets/9a53ec4e-3e9d-4dbf-8650-5a6d415128ea" width="100px" height="100px"/>

<img src="https://github.com/user-attachments/assets/307e832b-a932-4377-86a6-929f105e0e56" width="100px" height="100px"/>
<img src="https://github.com/user-attachments/assets/68841153-a802-40ef-a773-9490f66f7079" width="100px" height="100px"/>

<img src="https://github.com/user-attachments/assets/cc136e8a-78bd-480b-b052-9b16df175a16" width="100px" height="100px"/>
<img src="https://github.com/user-attachments/assets/4c3acb27-0d81-4f18-9e5c-c4cd89dbf1fd" width="100px" height="100px"/>

<br>
<img src="https://github.com/user-attachments/assets/b4fcf25f-5d26-4de5-8b5d-b831d6fb1e3e" width="300px" height="300px"/>

## Contato

Para mais informações sobre o projeto, favor entrar em contato com os autores por algum dos emails abaixo:

➡️ reginaldo.filho@ime.eb.br
<br>
➡️ matheus.silva@ime.eb.br
<br>
<br>
Mapear, Nobre Missão!
<br>
🛰️🇧🇷
