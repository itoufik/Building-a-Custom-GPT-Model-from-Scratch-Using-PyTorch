import pickle

class BasicTokenizer():

    def __init__(self , data_path : str = None , vocab_size : int = None , verbose:bool = False):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.verbose = verbose
        self.merges = {}
        self.vocab = {}


    def load_text(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return text
    
    
    @staticmethod
    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]): 
            counts[pair] = counts.get(pair, 0) + 1

        return counts 
    

    @staticmethod
    def merge(ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1

        return newids
    
        
    def train(self):
        assert self.vocab_size >= 256
        assert self.data_path is not None
        text = self.load_text()
        num_merges = self.vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if self.verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")
        print(f"compression ratio: {len(text_bytes) / len(ids):.2f}X")
        with open("merges.pkl" , "wb") as merge_file:
            pickle.dump(self.merges , merge_file)
        with open ("vocab.pkl" , "wb") as vf:
            pickle.dump(self.vocab , vf)


    def encode(self , text_to_encode):
        with open("merges.pkl" , "rb") as mf:
            merges = pickle.load(mf)
        text_bytes = text_to_encode.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            idx = merges[pair]
            ids = self.merge(ids, pair, idx)
        print(f"compression ratio: {len(text_bytes) / len(ids):.2f}X")

        return ids
    
    
    def decode(self , ids):
        with open("vocab.pkl" , "rb") as vf:
            vocab = pickle.load(vf)
        text_bytes = b"".join(vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        
        return text


