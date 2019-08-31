package cmd

import (
	"fmt"
	"github.com/fluhus/gostuff/nlp/wordnet"
	log "github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/config"
	"github.com/twatzl/imtag/tagger"
	"github.com/twatzl/imtag/tagger/image"
	"github.com/twatzl/imtag/tagger/imageClassifier"
	"github.com/twatzl/imtag/tagger/tag"
	"github.com/twatzl/imtag/tagger/word2vec"
	"github.com/twatzl/imtag/tagger/word2vec/skipGramModel"
	"os"
	"runtime"
	"strings"
)

// InitLogger inits a new logrus logger with the given
// log level.
func InitLogger(level log.Level) *log.Logger {
	logger := log.New()
	logger.SetReportCaller(true)
	logger.Formatter = &log.TextFormatter{
		CallerPrettyfier: func(f *runtime.Frame) (string, string) {
			repopath := fmt.Sprintf("%s/src/github.com/twatzl/imtag", os.Getenv("GOPATH"))
			filename := strings.Replace(f.File, repopath, "", -1)
			return fmt.Sprintf("%s()", f.Function), fmt.Sprintf("%s:%d", filename, f.Line)
		},
	}
	logger.SetLevel(level)
	return logger
}

// NewTaggerConfig will create a new tagger.TaggerConfig with the models given as parameters
// and the other values are taken from the config values.
// Models that are not needed may be set to nil.
func NewTaggerConfig(w2v word2vec.Word2Vec,
	wordnet *wordnet.WordNet,
	classifier imageClassifier.ImageClassifier,
	labelStorage tagger.LabelStorage) (conf tagger.TaggerConfig) {

	conf = tagger.TaggerConfig{
		K:                    config.GetK(),
		Confidence:           config.GetConfidence(),
		EmbedHierarchical:    config.HierarchicalEmbeddingEnabled(),
		RawClassifierResults: config.RawClassifierResultsEnabled(),
		Word2VecModel:        w2v,
		WordNet:              wordnet,
		ImageClassifier:      classifier,
		LabelStorage:         labelStorage,
	}
	return conf
}

func PrintResults(images []image.Image) {
	for _, i := range images {
		PrintImageResults(i)
	}
}

func PrintImageResults(i image.Image) {
	tags := i.GetTags()

	fmt.Printf("%s:\n", i.GetFilename())

	for _, ta := range tags {
		PrintTag(ta)
	}
}

func PrintTag(tag tag.Tag) {
	fmt.Printf("%s %f", tag.GetLabel(), tag.GetConfidence())
}
func LoadWord2VecModel(path string) (word2vec.Word2Vec, error) {
	// TODO: skipGramModel should be renamed to w2v. it just loads a pretrained model. no skipgram in here
	w2v := skipGramModel.New(path)

	// this should return an error in the future in case anything happens, however for this to work
	// we need to modify the new function.
	return w2v, nil
}

func LoadWordNet(path string) (wn *wordnet.WordNet, err error) {
	wn, err = wordnet.Parse(path)
	return wn, err
}
