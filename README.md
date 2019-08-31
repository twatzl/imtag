# imtag

A software for zero-shot tagging of images. imtag allows to tag images using labels for which no training data is available.

The basic principle is that the software uses a pre-trained classifier to classify an image. The resulting classes are then used
to embed the image in a n-dimensional vector space. For this embedding a word2vec model is needed for converting words into vectors
and a wordnet model is needed to be able to respect the hierarchy between different labels.

The pre-trained classifier can be obtained from the TensorFlow models repository.

## Usage

There are 3 commands for imtag. For detailed parameters please use `imtag <command> --help`.

### search

The `search` command helps the user finding out whether a given label is available in the word2vec model and the wordnet structure.

### addLabel

The `addLabel` command allows to register new labels with which images may be tagged.

### tag

The `tag` command can be used to tag given images.

**For now imtag only supports tagging a single image at a time.**

## Requirements

Additional data is required to run imtag. For licensing purposes this data cannot be supplied with imtag, but has to be downloaded and prepared by the use.

### TensorFlow Model

TODO add tutorial how to freeze a TensorFlow model.

### w2v

TODO add link

### wordnet

**Version 3.1**

```
mkdir -p data/wordnet3.1
cd data/wordnet3.1
wget http://wordnetcode.princeton.edu/wn3.1.dict.tar.gz
tar -xvf wn3.1.dict.tar.gz
```

**Version 3.0**

NOTE: wnid have changed between wn v3 and v3.1, so for evaluation you need WordNet 3.0

```
mkdir -p data/wordnet3.0
cd data/wordnet3.0
wget http://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz
tar -xvf WNdb-3.0.tar.gz
```


## Evaluation

For evaluation we used the following list of words [https://raw.githubusercontent.com/li-xirong/hierse/master/data/synsets_ilsvrc12_test1k_2hop.txt](https://raw.githubusercontent.com/li-xirong/hierse/master/data/synsets_ilsvrc12_test1k_2hop.txt)
