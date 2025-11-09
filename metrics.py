'''
we use the Porter Stemmer to stem the phrases, and after that using set() to remove duplicates.
'''
from nltk.stem import PorterStemmer
class MAP:
    def __init__(self,k):
        self.k = k
        self.MAP_sum = 0
        self.num = 0
        self.porter_stemmer = PorterStemmer()

    def compute_MAP(self,gold_labels,output_labels):
        self.num += 1
        precision_sum = 0
        length = min(len(output_labels),self.k)
        s_g_labels = stem_phrase_list(gold_labels,self.porter_stemmer)
        s_o_labels = stem_phrase_list(output_labels,self.porter_stemmer)
        
        for i in range(length):
            correct_labels = set(s_g_labels) & set(s_o_labels[:i+1])
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
        self.porter_stemmer = PorterStemmer()

        self.not_recall_labels = []
        self.gold_labels = []
        self.output_labels = []
        self.occr_correct_phrases = []
        self.not_occr_correct_phrases = []

    def compute_F1(self,gold_labels,output_labels):
        self.num += 1
        stemmed_gold_labels = stem_phrase_list(gold_labels,self.porter_stemmer)
        stemmed_output_labels = stem_phrase_list(output_labels[:self.k],self.porter_stemmer)
    
        correct_labels = set(stemmed_gold_labels) & set(stemmed_output_labels)
        correct_labels_c = len(correct_labels)       

        precision = correct_labels_c / self.k # If there is less than k generated labels, we fill the rest with empty strings
        recall = correct_labels_c / len(gold_labels)

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        self.precision_sum = self.precision_sum + precision
        self.recall_sum = self.recall_sum + correct_labels_c / len(gold_labels)
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
    @property
    def occr_ratio(self):
        if len(self.not_occr_correct_phrases) != 0:
            return len(self.occr_correct_phrases) / (len(self.occr_correct_phrases) + len(self.not_occr_correct_phrases))
        else:
            return -1

def stem_phrase_list(phrase_list,stemmer):
    stemmed_phrase_list = []
    for phrase in phrase_list:
        stemmed_phrase_list.append(stem_phrase(phrase,stemmer))
    return stemmed_phrase_list

def stem_phrase(phrase,stemmer):
    words = phrase.split(' ')
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return ' '.join(stemmed_words)

def compute_metrics(file_path,print_f = False):
    data = read_labels(file_path)  
    map_5 = MAP(5)
    f1_1 = MACRO_F1(1)
    f1_3 = MACRO_F1(3)
    for i in range(len(data)):
        row = data[i].split('<sep>')
        if len(row) == 1:
            continue
        gs = row[0].split(",")
        gs = [g.strip() for g in gs]
        os = row[1].split(",")
        os = [o.strip() for o in os]
        f1_1.compute_F1(gs, os)
        f1_3.compute_F1(gs, os)
        map_5.compute_MAP(gs, os)
    if print_f:
        print("F1@1: {:+.2f}%".format(f1_1.F1_score * 100))
        print("F1@1-precision: {:+.2f}%".format(f1_1.precision * 100))
        print("F1@1-recall: {:+.2f}%".format(f1_1.recall * 100))
        print("F1@3: {:+.2f}%".format(f1_3.F1_score * 100))
        print("F1@3-precision: {:+.2f}%".format(f1_3.precision * 100))
        print("F1@3-recall: {:+.2f}%".format(f1_3.recall * 100))
        print("MAP@5: {:+.2f}%".format(map_5.MAP_score * 100))
        print("Sum: {:+.2f}%".format(f1_1.F1_score * 100 + f1_3.F1_score * 100 + map_5.MAP_score * 100))
        return
    with open(file_path, 'a') as file:
        file.write("F1@1: {:+.2f}%".format(f1_1.F1_score * 100) + "\n")
        file.write("F1@1-precision: {:+.2f}%".format(f1_1.precision * 100) + "\n")
        file.write("F1@1-recall: {:+.2f}%".format(f1_1.recall * 100) + "\n")
        file.write("F1@3: {:+.2f}%".format(f1_3.F1_score * 100) + "\n")
        file.write("F1@3-precision: {:+.2f}%".format(f1_3.precision * 100) + "\n")
        file.write("F1@3-recall: {:+.2f}%".format(f1_3.recall * 100) + "\n")
        file.write("MAP@5: {:+.2f}%".format(map_5.MAP_score * 100) + "\n")
        file.write("Sum: {:+.2f}%".format(f1_1.F1_score * 100 + f1_3.F1_score * 100 + map_5.MAP_score * 100) + "\n")

def read_labels(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split('\n')
    labels = []
    for i in range(len(data)):
        if '<sep>' in data[i]:
            labels.append(data[i])
    return labels

if __name__ == "__main__":
    files = [
        "outputs path"
    ]
    for file in files:
        compute_metrics(file, print_f=True)