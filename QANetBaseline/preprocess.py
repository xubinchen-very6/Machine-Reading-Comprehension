import re
import jieba
from disk_io import load_json, load_lines

symbol_pat = re.compile(r'[\s|\n|*|\"|\'|【|】|\[|\]|（|）|\(|\)|“|”|、|《|》]+')
email_pat = re.compile(r'[\*]+@(.*)\.com')


def remove_useless(sentence):
    normalized = re.sub(symbol_pat, '', sentence)
    normalized = re.sub(email_pat, 'EMAIL', normalized)
    return normalized

def add_wrongspell_before_tokenize(spellcor_file):
    spell_dict = load_json(spellcor_file)
    for k in spell_dict.keys():                 
        jieba.suggest_freq(k, True)                         

def tokenize(userdict_file, spellcor_file):
    jieba.load_userdict(userdict_file)  
    add_wrongspell_before_tokenize(spellcor_file)
    jieba.initialize() 
    return lambda sentence: jieba.cut(sentence, cut_all=True)

def remove_sw_after_tokenize(stopwords_file):
    stopwords = set(load_lines(stopwords_file))
    return lambda l: list(filter(lambda w: w not in stopwords, l))


def preprocess_fn_init(userdict_file, spellcor_file, stopwords_file):
    tokenize_fn = tokenize(userdict_file, spellcor_file)
    remove_sw_fn = remove_sw_after_tokenize(stopwords_file)
    return lambda s: remove_sw_fn(tokenize_fn(remove_useless(s)))
