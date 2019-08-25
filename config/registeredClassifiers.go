package config

import (
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/tagger/imageClassifier"
	"github.com/twatzl/imtag/tagger/imageClassifier/tensorflowImageClassifier"
)



var tensorFlowFrozenModelInitializer = func (cd *ImageClassifierDesc, logger *logrus.Logger) (imageClassifier.ImageClassifier, error) {
	classifier := tensorflowImageClassifier.New(cd.dataPathMapper, logger)
	err := classifier.LoadFrozenModel(cd.Path())
	if err != nil {
		return nil, err
	}

	return classifier, nil
}

// knownImageClassifiers allows to refer to different classifiers using only a name instead of a path and type
// combination in the flags. This makes things easier for the user. The downside is that developers have to
// enter their classifiers here first.
var knownImageClassifiers = map[string]*ImageClassifierDesc{
	"VGG19": NewClassifierDesc(
		"VGG19",
		"frozen_vgg_19.pb",
		getCompletePathToData,
		tensorFlowFrozenModelInitializer,
	),
}

// GetClassifierModels returns a list of all classifiers that are implemented and
// registered.
func GetKnownClassifierModels() map[string]*ImageClassifierDesc {
	return knownImageClassifiers
}

