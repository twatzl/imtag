package imageClassifier

import "github.com/sirupsen/logrus"

type ClassifierFactory func(cd *ImageClassifierDesc, dataPath string, logger *logrus.Logger) (imageClassifier.ImageClassifier, error);

type ImageClassifierDesc struct {
	modelName       string
	modelPath       string
	instantiateFunc ClassifierFactory
}

func NewClassifierDesc(modelName, modelPath string, instantiateFunc ClassifierFactory) ImageClassifierDesc {
	return ImageClassifierDesc{
		modelName:       modelName,
		modelPath:       modelPath,
		instantiateFunc: instantiateFunc,
	}
}

func (cd *ImageClassifierDesc) Name() string {
	return cd.modelName
}

func (cd *ImageClassifierDesc) Path() string {
	return cd.modelName
}

func (cd *ImageClassifierDesc) InstantiateClassifier(dataPath string, logger *logrus.Logger) (ImageClassifier, error) {
	return cd.instantiateFunc(cd, dataPath, logger)
}
