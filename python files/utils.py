import unicodedata
import string
import re
import nltk

from nltk.stem import WordNetLemmatizer
from contractions import CONTRACTION_MAP
from nltk.corpus import stopwords
from nltk.corpus import wordnet


def remove_duplicates(text) -> str:
    """
       Function to remove duplicates
       :param: text from which duplicates need to be removed
       :return: string with no duplicates
    """
    words = text.split()
    return " ".join(sorted(set(words), key=words.index))


# reference from - https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
def remove_accents(text) -> str:
    """
          Function to remove accents characters from the string
          :param: string
          :return: string with no accented characters
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def remove_punc(text) -> str:
    """
              Function to remove punctuation from the text
              :param: text with punctuation
              :return: string with no punctuation
    """
    return ''.join(ch for ch in text if ch not in set(string.punctuation))


# reference from - https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
def remove_extra_whitespace_tabs(text) -> str:
    """
              Function to remove extra spaces tabs
              :param: text with extra white spaces
              :return: string with no extra white spaces
    """
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()


# code from - https://www.kdnuggets.com/2018/08/practitioners-guide-processing-understanding-text-2.html
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP) -> str:
    """
              Function to expand contractions based on the contraction map
              :param: string with contractions and contraction map
              :return: string with contractions expanded based on the map
    """
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_non_english_words(text) -> str:
    """
              Function to remove non english words
              :param: text with english and non english words
              :return: text with only english words
    """
    words = set(nltk.corpus.words.words())

    return " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())


def to_lowercase(text) -> str:
    """
              Function to lower case string
              :param: string
              :return: lower case words
    """
    return text.lower()


def to_uppercase(text) -> str:
    """
              Function to upper case string
              :param: string
              :return: upper case words
    """
    return text.upper()


# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
def remove_numbers(text) -> str:
    """
              Function to remove numbers from text
              :param: string
              :return: string with no numbers
    """
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
    return re.sub(pattern, '', text)


# Reference from - https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing
def remove_stopwords(text) -> str:
    """
              Function to remove stopwords
              :param: string with stopwords
              :return: string with stopwords removed
    """
    STOPWORDS = set(stopwords.words('english'))
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
def stemming(text) -> str:
    """
              Function to do stemming of the sentences
              :param: string
              :return: sentences with stemmed words
    """
    stemmer = nltk.porter.PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


# Reference from - https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing
def lemmatize_words(text):
    """
              Function to lemmatize the words
              :param: string
              :return: string with lemmatized words
    """
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join(
        [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
