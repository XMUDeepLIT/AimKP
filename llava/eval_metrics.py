from nltk.stem import PorterStemmer

def stem_list(list,stemmer):
    stemmed_list = []
    for phrase in list:
        stemmed_list.append(stem_phrase(phrase,stemmer))
    return stemmed_list

def stem_phrase(phrase,stemmer):
    words = phrase.split(' ')
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return ' '.join(stemmed_words)

class MAP:
    def __init__(self,k):
        self.k = k
        self.MAP_sum = 0
        self.num = 0

    def compute_MAP(self,labels,preds):
        self.num += 1
        precision_sum = 0
        length = min(len(preds),self.k)

        for i in range(length):
            correct_labels = set(labels) & set(preds[:i+1])
            correct_labels_c = len(correct_labels)
            precision_sum = precision_sum + correct_labels_c / (i + 1)

        if length != 0:
            map_score = precision_sum / length
        else:
            map_score = 0

        self.MAP_sum = self.MAP_sum + map_score

        return map_score

    @property
    def MAP_score(self):
        return self.MAP_sum / self.num
    
class MACRO_F1:
    def __init__(self,k):
        self.k = k
        self.precision_sum = 0
        self.recall_sum = 0
        self.F1_sum = 0
        self.num = 0

        self.not_recall_labels = []
        self.labels = []
        self.preds = []

    def compute_F1(self,labels,preds):
        self.num += 1
        preds = preds[:self.k]
        correct_labels = set(labels) & set(preds)
        correct_labels_c = len(correct_labels)       

        precision = correct_labels_c / self.k
        recall = correct_labels_c / len(labels)

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        self.precision_sum = self.precision_sum + precision
        self.recall_sum = self.recall_sum + correct_labels_c / len(labels)
        self.F1_sum = self.F1_sum + f1
        return f1,precision,recall
    @property
    def precision(self):
        return self.precision_sum / self.num
    @property
    def recall(self):
        return self.recall_sum / self.num
    @property
    def F1_score(self):
        return self.F1_sum / self.num
