package tagger

import (
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/tagger/image"
	"github.com/twatzl/imtag/tagger/tag"
	"fmt"
	"github.com/spf13/cobra"
)

func TagImage(cmd*cobra.Command, args []string) {
	logger := InitLogger(logrus.DebugLevel)

	config, err := LoadConfigFromCmd(cmd)
	tagger := New(config, logger)

	err = tagger.LoadClassifier(config.classifierName)
	if err != nil {
		// TODO error
	}
	err = tagger.LoadWord2VecModel(config.w2vName)
	if err != nil {
		// TODO error
	}

	path, err := config.GetImagePath()
	if err != nil {
		// TODO error
	}
	taggedImages, err := tagger.LoadAndTagImages(path)
	if err != nil {
		// TODO error
	}
	printResults(taggedImages)
}

func printResults(images []image.Image) {
	for _, i := range images {
		printImageResults(i)
	}
}

func printImageResults(i image.Image) {
	numTags := 10
	minConfidence := 0
	tags := i.GetTags()

	fmt.Printf("%s:\n", i.GetFilename())

	if minConfidence > 0 {
		for _, tag := range tags {
			printTag(tag)
		}
	} else {
		max := numTags
		if numTags > len(tags) {
			max = len(tags)
		}

		for i := 0; i < max; i++ {
			printTag(tags[i])
		}
	}
}

func printTag(tag tag.Tag) {
	fmt.Printf("%s %f", tag.GetLabel(), tag.GetConfidence())
}

//def cosine_similarity(vecx, vecy):
//norm = np.sqrt(np.dot(vecx, vecx))* np.sqrt(np.dot(vecy, vecy))
//return np.dot(vecx, vecy) / (norm + 1e-10)
