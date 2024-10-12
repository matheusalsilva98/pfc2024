# Elaboração de ferramenta para geração de mosaicos de imagens CBERS-4A livres de nuvens utilizando aprendizado profundo
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

![fluxograma-fase-treinamento](https://github.com/user-attachments/assets/2d01f869-d2d8-4fcd-95ec-60823ae9c4bc)

* Realizar os testes com os pesos salvos dos treinamentos realizados:

![fluxograma-fase-teste](https://github.com/user-attachments/assets/21650c50-2f52-4123-ac61-6f505f0a00ea)


[Acesse o link aqui](https://www.youtube.com/)

jaccard index | Acurácia | Precisão
:---: | :---: | :---:
0.9 | 0.89 | 0.98

Não entendo o comando `document.getElementById()`

Código em Python:

```
num = int('Digite um valor: ')
if num % 2 == 0:
  print(f'O valor {num} é PAR')
else:
  print(f'O valor {num} é ÍMPAR')
```

Exemplos de emoji 🇧🇷

Criando um quote
> Aqui está o quote!
