package config

import (
	"fmt"
	"github.com/pkg/errors"
	"github.com/spf13/viper"
	"os"
	"path"
)

/* Flag Names */
const FlagClassifierName = "classifier"
const FlagWord2VecModel = "w2v"
const FlagWordNetDictionary = "wordnet"
const FlagLabel = "label"
const FlagFile = "file"
const FlagK = "numResults"
const FlagConfidence = "confidence"
const FlagDataPath = "data"
const FlagRawClassifierResults = "rawClassification"
const FlagHierarchicalEmbedding = "hierarchicalEmbedding"

func InitConfigWithDefaultValues() {
	viper.SetDefault(FlagClassifierName, "VGG19")
	viper.SetDefault(FlagWord2VecModel, "./data/skipGram")
	viper.SetDefault(FlagWordNetDictionary, "./data/wordnet/dict")
	viper.SetDefault(FlagHierarchicalEmbedding, true)
	viper.SetDefault(FlagRawClassifierResults, false)
	viper.SetDefault(FlagK, 0)
	viper.SetDefault(FlagConfidence, 0)
	viper.SetDefault(FlagDataPath, "")
}

// VerifyConfigForEmbedLabel checks if all parameters needed for embedding a new label are set.
func VerifyConfigForEmbedLabel() (bool, []error) {
	// we want to display all errors to the user so he can fix all at once
	// TODO currently we do not try to load wordnet or w2v. maybe we should try that
	errorsFound := []error{}
	isValid := true

	if viper.GetString(FlagLabel) == "" {
		isValid = false
		err := errors.New("label must not be empty")
		errorsFound = append(errorsFound, err)
	}

	ok, err := IsWord2VecPathValid()
	if !ok {
		isValid = false
		errorsFound = append(errorsFound, err)
	}

	ok, err = IsWordNetPathValid()
	if !ok {
		isValid = false
		errorsFound = append(errorsFound, err)
	}

	return isValid, errorsFound
}

func VerifyConfigForTagImages() (bool, []error) {
	// we want to display all errors to the user so he can fix all at once
	// TODO currently we do not try to load wordnet or w2v. maybe we should try that
	errorsFound := []error{}
	isValid := true

	filePath := GetPathToImageFiles()

	info, err := os.Stat(filePath)
	if err != nil {
		isValid = false
		errorsFound = append(errorsFound, err)
	} else if !(info.Mode().IsDir() || info.Mode().IsRegular()) {
		isValid = false
		errorsFound = append(errorsFound, errors.New("given classifier path is neither file nor directory"))
	}

	if filePath == "" {
		isValid = false
		err := errors.New("label must not be empty")
		errorsFound = append(errorsFound, err)
	}

	ok, err := IsWord2VecPathValid()
	if !ok {
		isValid = false
		errorsFound = append(errorsFound, err)
	}

	if (HierarchicalEmbeddingEnabled()) {
		ok, err = IsWordNetPathValid()
		if !ok {
			isValid = false
			errorsFound = append(errorsFound, err)
		}
	}

	return isValid, errorsFound
}

func GetPathToImageFiles() string {
	return viper.GetString(FlagFile)
}

// IsClassifierNameValidAndExists checks if the given value for the classifier name is a valid name for one of the implemented
// classifiers and if the classifier file or directory does exists.
func IsClassifierNameValidAndExists() (bool, error) {
	classifierName := viper.GetString(FlagClassifierName)
	val, exists := GetKnownClassifierModels()[classifierName]
	if !exists {
		return false, errors.New(fmt.Sprintf("classifier %s does not exists", classifierName))
	}

	classifierPath := getCompletePathToData(val.Path())
	info, err := os.Stat(classifierPath)
	if err != nil {
		return false, err
	}

	if !(info.Mode().IsDir() || info.Mode().IsRegular()) {
		return false, errors.New("given classifier path is neither file nor directory")
	}

	return true, nil
}

func IsWord2VecPathValid() (bool, error) {
	w2vPath := GetWord2VecModelPath()

	info, err := os.Stat(w2vPath)
	if err != nil {
		return false, err
	}

	if !info.Mode().IsDir() {
		return false, errors.New("w2v model path must point to directory")
	}

	return true, nil
}

func IsWordNetPathValid() (bool, error) {
	wordnetPath := GetWordNetDictionaryPath()

	info, err := os.Stat(wordnetPath)
	if err != nil {
		return false, err
	}

	if !info.Mode().IsDir() {
		return false, errors.New("wordnet path must point to directory")
	}

	return true, nil
}

func GetClassifierDescription() *ImageClassifierDesc {
	classifierName := viper.GetString(FlagClassifierName)
	val, _ := GetKnownClassifierModels()[classifierName]
	return val
}

func GetWord2VecModelPath() string {
	return getCompletePathToData(viper.GetString(FlagWord2VecModel))
}

func GetWordNetDictionaryPath() string {
	return getCompletePathToData(viper.GetString(FlagWordNetDictionary))
}

func getCompletePathToData(itemPath string) (string) {
	if path.IsAbs(itemPath) {
		return itemPath
	}

	dataPath := viper.GetString(FlagDataPath)

	return path.Join(dataPath, itemPath)
}

func GetClassifierName() string {
	return viper.GetString(FlagClassifierName)
}

func HierarchicalEmbeddingEnabled() bool {
	return viper.GetBool(FlagHierarchicalEmbedding)
}

func RawClassifierResultsEnabled() bool {
	return viper.GetBool(FlagRawClassifierResults)
}

func GetK() int {
	return viper.GetInt(FlagK)
}

func GetConfidence() float64 {
	return viper.GetFloat64(FlagConfidence)
}