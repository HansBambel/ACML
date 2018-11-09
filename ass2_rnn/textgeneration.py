from textgenrnn import textgenrnn

textgen = textgenrnn(weights_path='weights/nips-abstracts_weights.hdf5',
                     vocab_path='weights/nips-abstracts_vocab.json',
                     config_path='weights/nips-abstracts_config.json')

textgen.generate(7, temperature=0.6)

# We consider the problem of user prediction has been proposed to achieve substantial descriptions in the weighted probabilistic model of the maximum likelihood estimator.
