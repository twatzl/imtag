package config

import (
	"errors"
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/tagger/imageClassifier"
)

type ClassifierFactory func(cd *ImageClassifierDesc, logger *logrus.Logger) (imageClassifier.ImageClassifier, error);

// DataPathMapper will add the data path to a resource name.
type DataPathMapper func(resourceName string) string

type ImageClassifierDesc struct {
	modelName       string
	modelPath       string
	dataPathMapper  DataPathMapper
	instantiateFunc ClassifierFactory
}

func NewClassifierDesc(modelName, modelPath string, dataPathMapper DataPathMapper, instantiateFunc ClassifierFactory) *ImageClassifierDesc {
	return &ImageClassifierDesc{
		modelName:       modelName,
		modelPath:       modelPath,
		dataPathMapper:  dataPathMapper,
		instantiateFunc: instantiateFunc,
	}
}

func (cd *ImageClassifierDesc) Name() string {
	return cd.modelName
}

func (cd *ImageClassifierDesc) Path() string {
	return cd.modelPath
}

func (cd *ImageClassifierDesc) InstantiateClassifier(logger *logrus.Logger) (imageClassifier.ImageClassifier, error) {
	if cd == nil {
		return nil, errors.New("classifier description is nil")
	}

	return cd.instantiateFunc(cd, logger)
}
