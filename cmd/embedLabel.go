package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	"github.com/twatzl/imtag/config"
	"github.com/twatzl/imtag/tagger"
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

	tc := NewTaggerConfig(nil, nil, nil)
	tagger := tagger.New(tc, logger)
	err := tagger.AddNewLabel(viper.GetString(config.FlagLabel))
	if err != nil {
		logger.WithError(err).Errorln("error when adding new label")
	}
}