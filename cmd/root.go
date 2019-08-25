// Copyright © 2019 Tobias Watzl

package cmd

import (
	"fmt"
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/config"
	"os"

	"github.com/mitchellh/go-homedir"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var cfgFile string

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "tagger",
	Short: "Tagger is meant to print tags for images.",
	Long: `Tagger is a zero shot image tagger, meaning it can tag images with labels for which there have been
no training samples provided before.`,
	// Uncomment the following line if your bare application
	// has an action associated with it:
	//	Run: func(cmd *cobra.Command, args []string) { },
}

var addLabelCmd = &cobra.Command{
	Use:   "addLabel",
	Short: "Register a new label for tagging.",
	Long: `addLabelCmd will register a new label for tagging. Based on similarity to other words it will be determined
if an image gets tagged with the label.`,
	Run: func(cmd *cobra.Command, args []string) {
		AddNewLabel()
	},
}

var tagCmd = &cobra.Command{
	Use:   "tag",
	Short: "Tag an image.",
	Long:  `tag will try to find matching labels for an image.`,
	Run: func(cmd *cobra.Command, args []string) {
		TagImage()
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)

	// Here you will define your flags and configuration settings.
	// Cobra supports persistent flags, which, if defined here,
	// will be global for your application.
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.test.yaml)")
	// general parameters
	rootCmd.PersistentFlags().String(config.FlagWord2VecModel,
		viper.GetString(config.FlagWord2VecModel),
		"The word2vec model to be used.")
	rootCmd.PersistentFlags().String(config.FlagWordNetDictionary,
		viper.GetString(config.FlagWordNetDictionary),
		"The wordnet dictionary to be used.")
	rootCmd.PersistentFlags().String(config.FlagDataPath,
		viper.GetString(config.FlagDataPath),
		"The path to where the data for the application is stored (i.e. mod els etc.)")
	rootCmd.PersistentFlags().Bool(config.FlagHierarchicalEmbedding,
		viper.GetBool(config.FlagHierarchicalEmbedding),
		"If this flag is set the embedding will take into account the whole wordnet hierarchy of the labels. "+
			"If the flag is not set only the label itself will be taken into account.")

	err := viper.BindPFlags(rootCmd.PersistentFlags())
	if err != nil {
		logrus.WithError(err).Errorln("could not bind flags for root cmd")
	}

	// parameters for embedding a new label
	addLabelCmd.Flags().StringP(config.FlagLabel, "l", "", "The label to register for tagging.")

	err = viper.BindPFlags(addLabelCmd.Flags())
	if err != nil {
		logrus.WithError(err).Errorln("could not bind flags for add label cmd")
	}

	// parameters for tagging
	tagCmd.Flags().StringP(config.FlagClassifierName, "c", viper.GetString(config.FlagClassifierName), "The classifier to be used for tagging.")
	tagCmd.Flags().StringP(config.FlagFile, "f", "", "The image file to tag")
	tagCmd.Flags().IntP(config.FlagK, "k", viper.GetInt(config.FlagK), "Will display the n most probable results with probability.")
	tagCmd.Flags().Float64P(config.FlagConfidence, "a", viper.GetFloat64(config.FlagConfidence), "Will display only tags with a confidence of more than c (c must be between 0 and 1). This will overridde -n flag.")

	err = viper.BindPFlags(tagCmd.Flags())
	if err != nil {
		logrus.WithError(err).Errorln("could not bind flags for tagging cmd")
	}

	rootCmd.AddCommand(addLabelCmd)
	rootCmd.AddCommand(tagCmd)
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {
	// set default values
	config.InitConfigWithDefaultValues()

	if cfgFile != "" {
		// Use config file from the flag.
		viper.SetConfigFile(cfgFile)
	} else {
		// Find home directory.
		home, err := homedir.Dir()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}

		// Search config in home directory with name "imtagConfig" (without extension).
		viper.AddConfigPath(home)
		// Search config also in current directory
		viper.AddConfigPath("./config")
		viper.SetConfigName("imtagConfig")
	}

	viper.AutomaticEnv() // read in environment variables that match

	// If a config file is found, read it in.
	if err := viper.ReadInConfig(); err == nil {
		fmt.Println("Using config file:", viper.ConfigFileUsed())
	}
}
