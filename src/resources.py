import os

# Some basic resource management
# Required folder structure:
# - project_root
#       - results               results
#       - data                  data
#           - embeddings        VSMs
#           - corpora           training and test data
#           - lexicons          lexicon lists

class ResourceManager:
    def __init__(self, root):
        self.root = root
        self.out = os.path.join(self.root, "results")
        self.data = os.path.join(self.root, "data")
        self.vsm_folder = os.path.join(self.data, "embeddings")
        self.corpora = os.path.join(self.data, "corpora")
        self.lexicons = os.path.join(self.data, "lexicons")

    def get_corpus(self, corpus_name):
        return (os.path.join(self.corpora, corpus_name+x) for x in [".sentences.conllx.flattened", ".frame.elements"])

    def get_lexicon(self, lexicon_name):
        return os.path.join(self.lexicons, lexicon_name) if lexicon_name is not None else None

    def get_vsm(self, vsm_name):
        return os.path.join(self.vsm_folder, vsm_name) if vsm_name is not None else None
