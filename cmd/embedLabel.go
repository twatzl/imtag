package cmd

import (
	"bufio"
	"github.com/fluhus/gostuff/nlp/wordnet"
	"github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	"github.com/twatzl/imtag/config"
	"github.com/twatzl/imtag/tagger"
	"os"
)

func AddNewLabel() {
	logger := InitLogger(logrus.DebugLevel)
	configValid, errors := config.VerifyConfigForEmbedLabel()

	if !configValid {
		for _, err := range errors {
			logger.WithError(err).Errorln("invalid configuration value")
		}
		return
	}

	var wn *wordnet.WordNet
	if config.HierarchicalEmbeddingEnabled() {
		var err error
		wn, err = LoadWordNet(config.GetWordNetDictionaryPath())
		if err != nil {
			logger.WithError(err).Errorln("could not load wordnet dictionary")
		}
	}

	ls := tagger.NewFileLabelStorage(logger,"./labelstore")
	err := ls.ReadFile()
	if err != nil {
		logger.Error("error during loading of label store")
		return
	}

	tc := NewTaggerConfig(nil, wn, nil, ls)
	imgTagger := tagger.New(tc, logger)

	label := viper.GetString(config.FlagLabel)
	labelFile := viper.GetString(config.FlagLabelFile)

	var labels []string
	if label != "" {
		labels = []string{label}
	}

	if labelFile != "" {
		file, err := os.Open(labelFile)
		if err != nil {
			logger.WithField("file", labelFile).WithError(err).Errorln("error opening file")
			return
		}

		defer file.Close()

		scanner := bufio.NewScanner(file)

		for scanner.Scan() {
			labels = append(labels, scanner.Text())
		}

		if err := scanner.Err(); err != nil {
			logger.WithField("file", labelFile).WithError(err).Errorln("error while reading file")
		}
	}

	for _, l := range labels {
		err := imgTagger.AddNewLabel(l)
		if err != nil {
			logger.WithField("label", l).WithError(err).Errorln("error when adding new label")
		}
	}

	err = ls.WriteFile()
	if err != nil {
		logger.Error("error during writing of label store")
		return
	}
}
