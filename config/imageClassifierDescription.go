package config

import (
	"errors"
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/tagger/imageClassifier"
	"github.com/twatzl/imtag/tagger/imageClassifier/gocvTfClassifier"
	"github.com/twatzl/imtag/tagger/imageClassifier/tensorflowImageClassifier"
)

type TensorFlowClassifierFactory func(cd *TensorFlowImageClassifierDesc, logger *logrus.Logger) (imageClassifier.ImageClassifier, error)
type GoCVClassifierFactory func(cd *GoCVClassifierDesc, logger *logrus.Logger) (imageClassifier.ImageClassifier, error)

// DataPathMapper will add the data path to a resource name.
type DataPathMapper func(resourceName string) string

type ImageClassifierDesc interface {
	Name() string
	Path() string
	InstantiateClassifier(logger *logrus.Logger) (imageClassifier.ImageClassifier, error)
}

type imageClassifierDesc struct {
	modelName       string
	modelPath       string
	dataPathMapper  DataPathMapper
}

type TensorFlowImageClassifierDesc struct {
	imageClassifierDesc
	instantiateFunc TensorFlowClassifierFactory
	classifierConfig tensorflowImageClassifier.Config
}

func NewTensorFlowClassifierDesc(
	modelName,
	modelPath string,
	dataPathMapper DataPathMapper,
	instantiateFunc TensorFlowClassifierFactory,
	classifierConfig tensorflowImageClassifier.Config) *TensorFlowImageClassifierDesc {
	return &TensorFlowImageClassifierDesc{
		imageClassifierDesc: imageClassifierDesc{
			modelName: modelName,
			modelPath:       modelPath,
			dataPathMapper:  dataPathMapper,
		},
		instantiateFunc: instantiateFunc,
		classifierConfig: classifierConfig,
	}
}

type GoCVClassifierDesc struct {
	imageClassifierDesc
	instantiateFunc GoCVClassifierFactory
	classifierConfig gocvTfClassifier.Config
}

func NewGoCVClassifierDesc(
	modelName,
	modelPath string,
	dataPathMapper DataPathMapper,
	instantiateFunc GoCVClassifierFactory,
	classifierConfig gocvTfClassifier.Config) *GoCVClassifierDesc {
	return &GoCVClassifierDesc{
		imageClassifierDesc: imageClassifierDesc{
			modelName: modelName,
			modelPath:       modelPath,
			dataPathMapper:  dataPathMapper,
		},
		instantiateFunc: instantiateFunc,
		classifierConfig: classifierConfig,
	}
}

func (cd *imageClassifierDesc) Name() string {
	return cd.modelName
}

func (cd *imageClassifierDesc) Path() string {
	return cd.modelPath
}

func (cd *TensorFlowImageClassifierDesc) InstantiateClassifier(logger *logrus.Logger) (imageClassifier.ImageClassifier, error) {
	if cd == nil {
		return nil, errors.New("classifier description is nil")
	}

	return cd.instantiateFunc(cd, logger)
}

func (cd *GoCVClassifierDesc) InstantiateClassifier(logger *logrus.Logger) (imageClassifier.ImageClassifier, error) {
	if cd == nil {
		return nil, errors.New("classifier description is nil")
	}

	return cd.instantiateFunc(cd, logger)
}
