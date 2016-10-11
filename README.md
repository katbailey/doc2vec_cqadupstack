Doc2Vec CQADupStack
====================

This repo provides code for reproducing the results of Jey Hau Lau and Timothy Baldwin in their paper, [An Empirical Evaluation of doc2vec with
Practical Insights into Document Embedding Generation](https://arxiv.org/pdf/1607.05368v1.pdf)

It currently only works with the dataset used for the forum question duplication task, CQADupStack. The dataset for this is available from http://nlp.cis.unimelb.edu.au/resources/cqadupstack/ and the script for processing it is included as a submodule of this repo.

There is one script in this repo, run.py, which allows you to perform all the necessary actions to extract a forum dataset, extract a small training set to work with (as described in the Lau paper), extract a test set of 10M documents, train a doc2vec model, and infer document vectors for all documents. It also allows you to infer document vectors using Lau et al's pretrained models from external corpora, Associated Press News and Wikipedia. These pretrained models are linked to from https://github.com/jhlau/doc2vec.

In addition to the pre-trained doc2vec models from external corpora, Lau et al created pretrained word2vec word embeddings from AP News and Wikipedia. These word embeddings are also linked to from the above github repo and can be used with this script to train new models.

Requirements
-------------
Lau et all [forked gensim](https://github.com/jhlau/gensim) to add the ability to train document vectors using pre-trained word embeddings. This repo provides a conda environment.yml file so you can create the environment needed to use this forked version of gensim.

Usage
------
To create the required python environment and activate it, run the following from the command line:
````
cd doc2vec_cqadup
conda env create -f environment.yml
source activate doc2vec
````

Assuming you have downloaded the zip files for the CQADupStack forum data to "path/to/cqadup/zip/files", to extract a particular forum dataset and run the train/test split provided by the CQADupStack script, run:
````
python run.py --name="english" --location="some/path" --cqadup-path="path/to/cqadup/zip/files" extract-dataset
````
This will place the extracted files for the "english" forum at "some/path". You will use this location for all of the extracted/processed files, including trained models and inferred vectors.

To extract a tiny training set of roughly 3000 negative and 300 positive examples, run:
```
python run.py --name="english" --location="some/path" extract-train-set
```

To extract a test set of 10M docs (using uniform random sampling, per the paper), run:
```
python run.py --name="english" --location="some/path" extract-test-set
```

To extract the text for all the documents into a file that has one document per line (as required by doc2vec), run:
````
python run.py --name="english" --location="some/path" extract-doc-text
```` 

To train a model from scratch using all of the documents:
````
python run.py --name="english" --location="some/path" train-model
````

This will result in a model.bin file being place at "some/path". To infer vectors for all the docs in the forum based on this model, run:
````
python run.py --name="english" --location="some/path" infer-doc-vectors
````

To infer vectors for some other set of documents based on this model, run:
````
python run.py --name="english" --location="some/path" --docs="path/to/some/docs/file" infer-doc-vectors
````
To use a different model, e.g. one of of the pretrained doc2vec models, to infer vectors for all the docs in the forum, run:
````
python run.py --name="english" --location="some/path" --model="path/to/pretrained/doc2vec/model/file" infer-doc-vectors
````

To train a model using pre-trained word-embeddings, the embeddings need to be in the non-binary word2vec format. The files linked to the from the Lau repo are not in this format but you can use the `convert-pretrained` command to convert them:
````
python run.py --words="path/to/apnews_sg/word2vec.bin" convert-pretrained
````
This will produce "path/to/apnews_sg/word2vec.txt", which can then be used to train new document vectors:
````
python run.py --words="path/to/apnews_sg/word2vec.txt" --name="english" --location="some/location" train-model
````

To use GloVe embeddings, these first need to be converted to word2vec format. Instead of `--words`, use the `--gloves` option when running convert-pretrained:
````
python run.py --gloves="path/to/glove/embeddings.txt" convert-pretrained
````
This will create a new file at "path/to/glove/embeddings.word2vec.txt", which can then be used to train a doc2vec model:
````
python run.py --words="path/to/glove/embeddings.word2vec.txt" --name="english" --location="some/location" train-model
````

The default number of iterations when training a model is 20, for inference it's 1000. For either command this can be overridden with the `--iter` option.
