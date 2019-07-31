package tagger

import (
	"github.com/spf13/cobra"
	"github.com/twatzl/imtag/cmd"
)

type TaggerConfig struct {
	command        *cobra.Command
	hierarchical   bool
	confidence     float32
	k              int
	dataPath       string
	classifierName string
	w2vName        string
}

func LoadConfigFromCmd(command *cobra.Command) (conf TaggerConfig, err error) {
	conf = TaggerConfig{
		command: command,
	}

	classifierName, err := command.Flags().GetString(cmd.FlagClassifier)
	if err != nil {
		// TODO
	}
	conf.classifierName = classifierName

	w2vName, err := command.Flags().GetString(cmd.FlagWord2VecModel)
	if err != nil {
		// TODO
	}
	conf.w2vName = w2vName

	k, err := command.Flags().GetInt(cmd.FlagK)
	if err != nil {
		// TODO
	}
	conf.k = k

	confidence, err := command.Flags().GetFloat32(cmd.FlagConfidence)
	if err != nil {
		// TODO
	}
	conf.confidence = confidence

	dataPath, err := command.Flags().GetString(cmd.FlagDataPath)
	if err != nil {
		// TODO
	}
	conf.dataPath = dataPath

	return conf, nil
}

func (tc *TaggerConfig) GetImagePath() (path string, err error) {
	imageFile, err := tc.command.Flags().GetString(cmd.FlagFile)
	if err != nil {
		// todo
		return "", err
	}

	return imageFile, nil
}
