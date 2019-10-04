package config

import (
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/tagger/imageClassifier"
	"github.com/twatzl/imtag/tagger/imageClassifier/gocvTfClassifier"
	"github.com/twatzl/imtag/tagger/imageClassifier/tensorflowImageClassifier"
	"gocv.io/x/gocv"
)

var tensorFlowFrozenModelInitializer = func(cd *TensorFlowImageClassifierDesc, logger *logrus.Logger) (imageClassifier.ImageClassifier, error) {
	classifier := tensorflowImageClassifier.New(cd.dataPathMapper, logger, cd.classifierConfig)
	err := classifier.LoadFrozenModel(cd.Path())
	if err != nil {
		return nil, err
	}

	return classifier, nil
}

var gocvTensorFlowModelInitializer = func(cd *GoCVClassifierDesc, logger *logrus.Logger) (imageClassifier.ImageClassifier, error) {
	classifier := gocvTfClassifier.New(cd.dataPathMapper, logger, cd.classifierConfig)
	err := classifier.LoadFrozenModel(cd.Path())
	if err != nil {
		return nil, err
	}

	return classifier, nil
}

// knownImageClassifiers allows to refer to different classifiers using only a name instead of a path and type
// combination in the flags. This makes things easier for the user. The downside is that developers have to
// enter their classifiers here first.
var knownImageClassifiers = map[string]ImageClassifierDesc{
	"VGG19": NewTensorFlowClassifierDesc(
		"VGG19",
		"frozen_vgg_19.pb",
		getCompletePathToClassifier,
		tensorFlowFrozenModelInitializer,
		tensorflowImageClassifier.Config{
			InputTag:  "input",
			OutputTag: "vgg_19/fc8/squeezed",
			LabelFile: "imagenet_comp_graph_label_strings.txt",
			NumLabels: 1000,
		},
	),
	"resnet_v2_152": NewTensorFlowClassifierDesc(
		"resnet_v2_152",
		"frozen_resnet_v2_152.pb",
		getCompletePathToClassifier,
		tensorFlowFrozenModelInitializer,
		tensorflowImageClassifier.Config{
			InputTag:  "input",
			OutputTag: "resnet_v2_152/predictions/Reshape_1",
			LabelFile: "imagenet_comp_graph_label_strings.txt",
			NumLabels: 1001,
		},
	),
	"gocv_resnet": NewGoCVClassifierDesc(
		"resnet_v2_152",
		"frozen_resnet_v2_152.pb",
		getCompletePathToClassifier,
		gocvTensorFlowModelInitializer,
		gocvTfClassifier.Config{
			InputTag:  "input",
			OutputTag: "resnet_v2_152/predictions/Reshape_1",
			LabelFile: "imagenet_comp_graph_label_strings.txt",
			NumLabels: 1001,
			Backend: gocv.NetBackendOpenCV,
			Target: gocv.NetTargetCPU,
		},
	),
	"gocv_vgg": NewGoCVClassifierDesc(
		"VGG19",
		"frozen_vgg_19.pb",
		getCompletePathToClassifier,
		gocvTensorFlowModelInitializer,
		gocvTfClassifier.Config{
			InputTag:  "input",
			OutputTag: "vgg_19/fc8/squeezed",
			LabelFile: "imagenet_comp_graph_label_strings.txt",
			NumLabels: 1000,
			Backend: gocv.NetBackendOpenCV,
			Target: gocv.NetTargetCPU,
		},
	),
}

// GetClassifierModels returns a list of all classifiers that are implemented and
// registered.
func GetKnownClassifierModels() map[string]ImageClassifierDesc {
	return knownImageClassifiers
}

func GetKnownClassifierNames() []string {
	var names []string
	for k, _ := range knownImageClassifiers {
		names = append(names, k)
	}
	return names
}
