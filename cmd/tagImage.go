package cmd

import (
	"github.com/fluhus/gostuff/nlp/wordnet"
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/config"
	"github.com/twatzl/imtag/tagger"
)

func TagImage() {
	logger := InitLogger(logrus.DebugLevel)
	configValid, errors := config.VerifyConfigForTagImages()

	if !configValid {
		for _, err := range errors {
			logger.WithError(err).Errorln("invalid configuration value")
		}
		return
	}

	w2v, err := LoadWord2VecModel(config.GetWord2VecModelPath())
	if err != nil {
		logger.WithError(err).Errorln("could not load word2vec model")
		return
	}

	var wn *wordnet.WordNet
	if config.HierarchicalEmbeddingEnabled() {
		wn, err = LoadWordNet(config.GetWordNetDictionaryPath())
		if err != nil {
			logger.WithError(err).Errorln("could not load wordnet dictionary")
		}
	}

	classifier, err := config.GetClassifierDescription().InstantiateClassifier(logger)
	if err != nil {
		logger.WithError(err).Errorln("could not load image classifier")
	}

	tc := NewTaggerConfig(w2v, wn, classifier)
	t := tagger.New(tc, logger)

	taggedImages, err := t.LoadAndTagImages(config.GetPathToImageFiles())
	if err != nil {
		// TODO error
	}
	PrintResults(taggedImages)
}