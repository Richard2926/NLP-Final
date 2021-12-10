import math, random
from typing import List, Tuple

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return ['~'] * n

Pair = Tuple[str, str]
Ngrams = List[Pair]
def ngrams(n, text:str) -> Ngrams:
    text=text.strip().split()
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    curr = n 
    modText = start_pad(n) + text
    # print(modText)
    grams = []
    for i in range(n, len(modText)):
      currstring = ""
      for j in range(i - n, i):
        currstring = currstring + modText[j]
        if(j < i - 1):
          currstring = currstring + " "
      if((currstring, modText[i])) != None:
        grams.append((currstring, modText[i]))
    return grams
    pass

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.contexts = set()
        self.ngrams = {}
        self.count = {}

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        return self.vocab
        pass

    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        grams = ngrams(self.n, text)
        for item in grams:
          if item in self.ngrams:
            # first, second = item
            self.ngrams[item] = self.ngrams[item] + 1
          else:
            self.ngrams[item] = 1
          if item[0] in self.count:
            self.count[item[0]] = self.count[item[0]] + 1
          else:
            self.count[item[0]] = 1
          self.vocab.add(item[1]) 
          self.contexts.add(item[0])
        pass

    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''
        gram = (context,word)
        if context not in self.contexts:
          return 1/len(self.vocab)
        if gram not in self.ngrams:
          x = 0
        else:
          x = self.ngrams[gram]
        if(self.k != 0):
          return ((x + self.k)/(self.count[context] + self.k * len(self.vocab)))
        else:
          return(x/self.count[context])
        ''' Returns the probability of char appearing after context '''
        pass

    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
#         random.seed(1)
        r = random.random()
        total = 0.0
        thisvoc = list(self.vocab)
        thisvoc.sort()
        # print(thisvoc)
        # print(thisvoc)
        for char in thisvoc:
          total = total + self.prob(context,char)
          if(total > r):
            return char
        pass

    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        strn = ["~"] * self.n
        # print(strn)
        returned = ""
        currstr = ""
        for i in range(0+self.n, length+self.n):
          # currstr = "hi"
          # for j in range(i-self.n, i):
            # print(j)
            # print(i)
            # print(strn[j])
            # print(currstr)
            # currstr = currstr+strn[j]
          # print(currstr)
          # print(strn) 
          # strn = strn.append(self.random_word(currstr))
          currCont = ""
          for j in range(i-self.n, i):
            currCont = currCont + strn[j] + " "
          # print(currCont[:-1])
          currCont = currCont[:-1]
          word = self.random_word(currCont)
          strn.append(word)
          returned = returned +  word + " "
          # print(currCont)
          # print(strn)
        # strn = strn[(2 * self.n - 1):]
        return returned[0:len(returned)-1]
        pass

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        w = 0
        grams = ngrams(self.n, text)
        for gram in grams:
          if(self.prob(gram[0], gram[1]) == 0.0):
            return float('inf')
          w = w + math.log(1/(self.prob(gram[0], gram[1])))
        return math.exp(w/len(grams))
        pass

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n 
        self.k = k 
        # self.vocab = set()
        # self.count = {}
        self.lamb = [1/(n+1)] * (n+1)
        self.models = [NgramModel(i,k) for i in range(n+1)]
        pass

    def get_vocab(self):
        return self.models[0].get_vocab()
        pass

    def update(self, text:str):
        for model in self.models:
          model.update(text)
        pass

    def prob(self, context:str, word:str):
      contextSplit = context.split()
      p = 0
      for i, val in enumerate(self.lamb):
        cont = ''
        if i is not 0:
          for j in range(0, i):
            cont = contextSplit[len(contextSplit) - j - 1] + " " + cont
          cont = cont[:-1]
        p = p + (val * self.models[i].prob(cont, word))
      return p

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass