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

TODO add link

