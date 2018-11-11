from textgenrnn import textgenrnn
import numpy as np

textgen = textgenrnn(weights_path='weights/nips-abstracts_weights.hdf5',
                     vocab_path='weights/nips-abstracts_vocab.json',
                     config_path='weights/nips-abstracts_config.json')
creativity = np.linspace(0.1, 1.5, 15)
# print(creativity)
for c in creativity:
    print(f'##### Creativity: {c} #####')
    textgen.generate(3, temperature=c)

# We consider the problem of user prediction has been proposed to achieve substantial descriptions in the weighted probabilistic model of the maximum likelihood estimator.
