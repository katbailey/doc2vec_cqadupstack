from __future__ import print_function
import sys
sys.path.append("CQADupStack")
import query_cqadupstack as qcqa
import pickle
import numpy as np
import math
from optparse import OptionParser
import gensim.models as g
import logging
import codecs
import os

def extract_dataset(name, path_to_zip, output_path):
    print("Loading the %s subforum" % name)
    src = qcqa.load_subforum("%s/%s.zip" % (path_to_zip, name))
    current_path = os.getcwd()
    if not output_path.startswith("/"):
        output_path = "%s/%s" % (current_path, output_path)
    # Remove trailing "/"
    output_path = output_path.rstrip("/")
    try:
        os.stat(output_path)
    except:
        try:
            os.mkdir(output_path)
        except:
            print("output path %s could not be created" % output_path)
            exit(-1)

    full_path = "%s/%s" % (output_path, name)
    try:
        os.stat(full_path)
    except:
        try:
            os.mkdir(full_path)
        except:
            print("output path %s could not be created" % full_path)
            exit(-1)
    output = open("%s/%s_src.pkl" % (full_path, name), 'w+')
    pickle.dump(src, output)
    output.close()

    try:
        os.chdir(full_path)
        print("Splitting %s for classification into output path %s" % (name, full_path))
        src.split_for_classification()
    except:
        print("Could not access output path")
        exit(-1)
    os.chdir(current_path)

def extract_train_set(name, location):
    output_file = "%s/%s/%s_trainpairs_tiny.txt" % (location, name, name)
    fp = open(output_file, "w+")
    # The probability of a pair being a duplicate is 0.0000022
    p = 0.0000022
    for i in range(10):
        fname = "%s/%s/%s_trainpairs_%s.txt" %(location, name, name, str(i + 1).zfill(2))
        with open(fname) as f:
            lines = f.readlines()
            num_total = len(lines)
            num_pos = math.floor(num_total * p) # This is the *expected* number of positive examples
            print("Processing file %s" % fname)
            # We want to sample roughly 30 positive examples from each file. If there are fewer
            # than 30 we'll take them all, otherwise we'll take each one with a probability
            # of 30 / num_pos
            sample_prob_pos = 1.0
            if num_pos > 30:
                sample_prob_pos = 30/num_pos
            # We want to sample at most 300 negative lines from each file
            sample_prob_neg = 300/(num_total - num_pos)
            for line in lines:
                line = line.strip()
                if line.endswith("1"):
                    if sample_prob_pos == 1.0 or np.random.sample() <= sample_prob_pos:
                        fp.write(line + "\n")
                else:
                    my_sample = np.random.sample()
                    if my_sample <= sample_prob_neg:
                        fp.write(line + "\n")
    fp.close()   
    print("Created train set at %s" % output_file)


def extract_test_set(name, location):
    output_file = "%s/%s/%s_testpairs_med.txt" % (location, name, name)
    fp = open(output_file, "w+")
    fname = "%s/%s/%s_testpairs_large.txt" % (location, name, name)
    print("Sampling 10M documents from %s" % fname)
    with open(fname) as f:
        pairs = []
        for line in f:
            pairs.append(line)
        inds = np.random.choice(len(pairs), 10000000, replace=False)
        pairs_arr = np.array(pairs)
        filtered = pairs_arr[inds]
        fp.writelines(filtered)
    fp.close()
    print("Created test set at %s" % output_file)

def train_model(name, location, docs_file = None, pretrained_emb = None, vector_size = 300, window_size = 15, min_count = 5, sampling_threshold = 1e-5, negative_size = 5, train_epoch = 20, dm = 0, worker_count = 1):
    if name == None or location == None:
        if docs_file == None:
            raise Exception("No training corpus provided!")
        train_corpus = docs_file
        output_file = "%s.model.bin" % docs_file
    else:
        train_corpus = "%s/%s/%s_docs.txt" % (location, name, name)
        output_file = "%s/%s/model.bin" % (location, name)
    #enable logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #train doc2vec model
    docs = g.doc2vec.TaggedLineDocument(train_corpus)
    model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=0, pretrained_emb=pretrained_emb, iter=train_epoch)
    #save model
    model.save(output_file)

def infer_doc_vectors(name = None, location = None, model_file = None, docs_file = None, start_alpha = 0.01, infer_epoch = 1000):
    #load model
    if name == None or location == None:
        if docs_file == None or model_file == None:
            raise Exception("No vectors to infer!")
        output_file = "%s.vectors.txt" % docs_file
    else:
        output_file = "%s/%s/%s_vectors.txt" % (location, name, name)
        if model_file == None:
            model_file = "%s/%s/model.bin" % (location, name)
        if docs_file == None:
            docs_file = "%s/%s/%s_docs.txt" % (location, name, name)
    m = g.Doc2Vec.load(model_file)
    test_docs = [ x.strip().split() for x in codecs.open(docs_file, "r", "utf-8").readlines() ]
    print("Inferring %s vectors (%s epochs)" % (len(test_docs), infer_epoch))
    #infer test vectors
    output = open(output_file, "w+")
    for i,d in enumerate(test_docs):
        joined_vector = " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)])
        output.write( joined_vector + "\n" )
    output.flush()
    output.close()

def extract_all_docs(name, location):
    pkl_file = open("%s/%s/%s_src.pkl" % (location, name, name), 'rb')
    src = pickle.load(pkl_file)
    pkl_file.close()
    output_file = "%s/%s/%s_docs.txt" % (location, name, name)
    fp = open(output_file, "w+")
    post_ids = src.get_all_postids()
    print("Extracting %s documents" % len(post_ids))
    all_docs = {}
    for pid in post_ids:
       all_docs[pid] = src.perform_cleaning(src.get_posttitle(pid), remove_punct=True) + " " + src.perform_cleaning(src.get_postbody(pid), remove_punct=True)

    mapping = {}
    for i,k in enumerate(all_docs.keys()):
        mapping[k] = i
        fp.write(all_docs[k].encode('utf-8') + "\n")
    fp.close()
    # Write the mapping file as a pickled object
    mp = open("%s/%s/%s_mapping.pkl" % (location, name, name), 'wb')
    pickle.dump(mapping, mp)
    mp.close()
    print("Extracted all documents to %s" % output_file)

def convert_word2vec_bin_to_nonbin(binary_words_file):
    fname, ext = os.path.splitext(binary_words_file)
    if not ext == ".bin":
        raise Exception("Pass a binary file of word embeddings")
    output_file = "%s.txt" % fname
    model = g.Word2Vec.load(binary_words_file)
    print("Model loaded, saving non-binary version to %s" % output_file)
    model.save_word2vec_format(output_file, binary=False)

def convert_glove_to_word2vec(glove_embeddings_file):
    fname, ext = os.path.splitext(glove_embeddings_file)
    output_fname = "%s.word2vec.txt" % fname
    output_file = open(output_fname, "wb")

    with open(glove_embeddings_file, "rb") as f:
        lines = f.readlines()
        num_lines = len(lines)
        num_dims = len(lines[0].split()) - 1
        gensim_first_line = "{} {}".format(num_lines, num_dims)
        output_file.write(gensim_first_line + "\n")
        for line in lines:
            output_file.write(line)
    output_file.close()
    print("Created word2vec version of glove file at %s" % output_fname)

if __name__ == '__main__':
    usage = "usage: %prog [options] command"
    parser = OptionParser(usage=usage)
    parser.add_option("-n", "--name", dest="name", help="Specify which forum to work with, e.g. \"gis\", \"tex\", etc")
    parser.add_option("-l", "--location", dest="location", help="Specify the location for extracted and processed files")
    parser.add_option("-c", "--cqadup-path", dest="cqadup", help="Path to where the cqadup zip files are located")
    parser.add_option("-i", "--iter", dest="num_iter", help="Number of iterations")
    parser.add_option("-m", "--model", dest="model", help="Specify a pre-trained model to use for inferring vectors")
    parser.add_option("-d", "--docs", dest="docs", help="Specify a file containing documents to infer vectors for")
    parser.add_option("-w", "--words", dest="word_embeddings", help="Specify a file containing pre-trained word embeddings to use")
    parser.add_option("-g", "--gloves", dest="gloves", help="Specify a file containing pre-trained GloVe word embeddings to convert to word2vec format")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        exit(-1)

    if args[0] == "extract-dataset" and type(options.name) != type(None) and type(options.location) != type(None) and type(options.cqadup) != type(None):
        extract_dataset(options.name, options.cqadup, options.location)
    elif args[0] == "extract-train-set" and type(options.name) != type(None) and type(options.location) != type(None):
        extract_train_set(options.name, options.location)
    elif args[0] == "extract-test-set" and type(options.name) != type(None) and type(options.location) != type(None):
        extract_test_set(options.name, options.location)
    elif args[0] == "train-model":
        d = {}
        if type(options.name) != type(None) and type(options.location) != type(None):
            d['name'] = options.name
            d['location'] = options.location
        elif type(options.docs) != type(None):
            d['docs_file'] = options.docs
        else:
            print("You must either specify name and loation or a docs file to train a model")
            exit(-1)
        if type(options.num_iter) != type(None):
            d['train_epoch'] = int(options.num_iter)
        if type(options.word_embeddings) != type(None):
            d['pretrained_emb'] = options.word_embeddings
        train_model(options.name, options.location, **d)
    elif args[0] == "infer-doc-vectors":
        d = {}
        if type(options.name) != type(None) and type(options.location) != type(None):
            d['name'] = options.name
            d['location'] = options.location
        elif type(options.docs) != type(None) and type(options.model) != type(None):
            d['docs_file'] = options.docs
            d['model_file'] = options.model
        else:
            print("You must either specify name and loation or a docs file and model for inferring vectors")
            exit(-1)
        if type(options.model) != type(None):
            d['model_file'] = options.model
        if type(options.num_iter) != type(None):
            d['infer_epoch'] = int(options.num_iter)
        infer_doc_vectors(**d)
    elif args[0] == "extract-doc-text" and type(options.name) != type(None) and type(options.location) != type(None):
        extract_all_docs(options.name, options.location)
    elif args[0] == "convert-pretrained":
        if type(options.word_embeddings) != type(None):
            convert_word2vec_bin_to_nonbin(options.word_embeddings)
        elif type(options.gloves) != type(None):
            convert_glove_to_word2vec(options.gloves)
    else:
        parser.print_help()
