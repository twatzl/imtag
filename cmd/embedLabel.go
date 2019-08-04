package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	"github.com/twatzl/imtag/config"
)

func EmbedLabel() {
	logger := InitLogger(logrus.DebugLevel)
	configValid, errors := config.VerifyConfigForTagImages()

	if !configValid {
		for _, err := range errors {
			logger.WithError(err).Errorln("invalid configuration value")
		}
		return
	}

	tc := NewTaggerConfig()
	tagger := New(tc, logger)
	err := tagger.EmbedNewLabel(viper.GetString(config.FlagLabel))
	if err != nil {
		logger.WithError(err).Errorln("error during embedding of new label")
	}
}