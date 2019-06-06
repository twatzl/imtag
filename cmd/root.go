// Copyright © 2019 Tobias Watzl

package cmd

import (
	"fmt"
	"os"

	homedir "github.com/mitchellh/go-homedir"
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
	Use: "addLabel",
	Short: "Register a new label for tagging.",
	Long: `addLabelCmd will register a new label for tagging. Based on similarity to other words it will be determined
if an image gets tagged with the label.`,
	Run: func(cmd *cobra.Command, args []string) {

	},
}

var tagCmd = &cobra.Command{
	Use: "tag",
	Short: "Tag an image.",
	Long: `tag will try to find matching labels for an image.`,
	Run: func(cmd *cobra.Command, args []string) {
		
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

	// Cobra also supports local flags, which will only run
	// when this action is called directly.
	rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")

	addLabelCmd.Flags().StringP("label", "l", "","The label to register for tagging.")

	tagCmd.Flags().StringP("file", "f", "", "The image file to tag")
	tagCmd.Flags().IntP("numResults", "n", 10, "Will display the n most probable results with probability.")
	tagCmd.Flags().Float32P("confidence", "c", 0, "Will display only tags with a confidence of more than c (c must be between 0 and 1). This will overridde -n flag.")

	rootCmd.AddCommand(addLabelCmd)
	rootCmd.AddCommand(tagCmd)
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {
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

		// Search config in home directory with name ".test" (without extension).
		viper.AddConfigPath(home)
		viper.SetConfigName(".test")
	}

	viper.AutomaticEnv() // read in environment variables that match

	// If a config file is found, read it in.
	if err := viper.ReadInConfig(); err == nil {
		fmt.Println("Using config file:", viper.ConfigFileUsed())
	}
}
