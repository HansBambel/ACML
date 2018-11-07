#  https://www.kaggle.com/jrobischon/wikipedia-movie-plots/home
#  https://www.kaggle.com/benhamner/nips-papers
#  https://github.com/minimaxir/textgenrnn
#  https://minimaxir.com/2018/05/text-neural-networks/
#  My colaboratory: https://colab.research.google.com/drive/17LVixGtYNgBX6XBbWThumXnZXeAr8K_h#scrollTo=aeXshJM-Cuaf

import pandas
import numpy as np

# df = pandas.read_csv('wiki_movie_plots_deduped.csv')
# # print(df['Plot'])
# with open('movie_plots.txt', 'w', encoding='utf-8') as outputfile:
#     for m in df['Plot']:
#         outputfile.write(m)

df = pandas.read_csv('nips_papers.csv')
# print(df['Plot'])
with open('nips_papers_abstract.txt', 'w', encoding='utf-8') as outputfile:
    for m in df['abstract']:
        if m != 'Abstract Missing':
            outputfile.write(m)
