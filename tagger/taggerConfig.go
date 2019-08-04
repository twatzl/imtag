package tagger

import (
	"github.com/fluhus/gostuff/nlp/wordnet"
	"github.com/twatzl/imtag/tagger/imageClassifier"
	"github.com/twatzl/imtag/tagger/word2vec"
)

type TaggerConfig struct {
	Confidence      float64
	K               int
	Word2VecModel   word2vec.Word2Vec
	WordNet         *wordnet.WordNet
	ImageClassifier imageClassifier.ImageClassifier
}