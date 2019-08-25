package cmd

import (
	"github.com/fluhus/gostuff/nlp/wordnet"
	"github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	"github.com/twatzl/imtag/config"
)

func SearchLabel() {
	logger := InitLogger(logrus.DebugLevel)
	label := viper.GetString("label")

	logger.WithField("label", label).Infoln("looking for label in word2vec and wordnet")

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

	vec := w2v.Word2Vec(label)
	if vec == nil {
		logger.Warnln("label not known to word2vec implementation")
	} else {
		logger.Infoln("label found in word2vec")
	}

	synset := wn.Search(label)["n"]
	if synset == nil {
		logger.Warnln("label is not in wordnet")
	} else {
		logger.WithField("synsetid", synset[0].Id()).WithField("synsetdesc", synset[0].String()).Infoln("label found in wordnet")
	}

}
