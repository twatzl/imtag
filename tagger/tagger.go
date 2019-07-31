package tagger

import (
	"fmt"
	"github.com/mewkiz/pkg/osutil"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/twatzl/imtag/cmd"
	"github.com/twatzl/imtag/tagger/image"
	"github.com/twatzl/imtag/tagger/imageClassifier"
	"github.com/twatzl/imtag/tagger/imageClassifier/tensorflowImageClassifier"
	"github.com/twatzl/imtag/tagger/tag"
	"github.com/twatzl/imtag/tagger/word2vec"
	"github.com/twatzl/imtag/tagger/word2vec/skipGramModel"
	"os"
)

type TensorFlowModelType int;

var tensorFlowFrozenModelInitializer = func (cd *imageClassifier.ImageClassifierDesc,dataPath string, logger *logrus.Logger) (imageClassifier.ImageClassifier, error) {
	classifier := tensorflowImageClassifier.New(dataPath, logger)
	err := classifier.LoadFrozenModel(cd.Path())
	if err != nil {
		return nil, err
	}

	return classifier, nil
}

var classifierModels = []imageClassifier.ImageClassifierDesc{
	imageClassifier.NewClassifierDesc(
		"VGG19",
		"frozen_vgg_19.pb",
		tensorFlowFrozenModelInitializer,
		),
}

type Tagger interface {
	LoadWord2VecModel(name string) error
	LoadClassifier(name string) error
	EmbedNewLabel() error
	LoadAndTagImages(imagePath string) ([]image.Image, error)
}

type tagger struct {
	imageClassifier imageClassifier.ImageClassifier
	word2vec        word2vec.Word2Vec
	logger          *logrus.Logger
	conf            TaggerConfig
}

func New(config TaggerConfig, logger *logrus.Logger) Tagger {
	tagger := &tagger{
		logger: logger,
		conf:   config,
	}

	return tagger
}

func (t *tagger) EmbedNewLabel() error {
	// TODO: embed the new label somewhere and then save the resulting vector
	panic("implement me")
}

func (t *tagger) LoadAndTagImages(imagePath string) (result []image.Image, err error) {

	if t.imageClassifier == nil {
		err = errors.New("no classifier loaded")
		return nil, err
	}

	if t.word2vec == nil {
		err = errors.New("no word2vec model loaded")
		return nil, err
	}

	images, err := t.prepareImageBatch(imagePath)
	if err != nil {
		return nil, err
	}

	var tags [][]tag.Tag
	if t.conf.k == 0 {
		tags = t.imageClassifier.ClassifyImages(images)
	} else {
		tags = t.imageClassifier.ClassifyImagesTopK(images, t.conf.k)
	}

	if tags == nil {
		err = errors.New("got no tags for image")
		return nil, err
	}

	for i, img := range images {
		img.SetTags(tags[i])
	}

	return images, nil
}

func (t *tagger) LoadClassifier(name string) error {
	var classifierDesc *imageClassifier.ImageClassifierDesc = nil

	for _, c := range classifierModels {
		if c.Name() == name {
			classifierDesc = &c;
			break;
		}
	}

	if classifierDesc == nil {
		err := errors.New(fmt.Sprintf("classifier with name %s not found", classifierModels))
		return err
	}

	classifier, err := classifierDesc.InstantiateClassifier(t.conf.dataPath, t.logger)
	if err != nil {
		// TODO
	}
	t.imageClassifier = classifier
}

func (t *tagger) LoadWord2VecModel(s string) error {
	// TODO: skipGramModel should be renamed to w2v. it just loads a pretrained model. no skipgram in here
	w2v := skipGramModel.New(s)
	t.word2vec = w2v

	// this should return an error in the future in case anything happens, however for this to work
	// we need to modify the new function.
	return nil
}

func (t *tagger) prepareImageBatch(imagePath string) (imageBatch []image.Image, err error){

	if !osutil.Exists(imagePath) {
		message := "path does not exist"
		t.logger.WithField("imagePath", imagePath).Errorln(message)
		err = errors.New(message)
		return nil, err
	}

	fi, err := os.Stat(imagePath)
	if err != nil {
		message := "could not stat path"
		t.logger.WithField("imagePath", imagePath).Errorln(message)
		err = errors.New(message)
		return nil, err
	}

	var images []image.Image = nil
	if fi.Mode().IsDir() {
		images = t.imageBatch(imagePath)
	} else if fi.Mode().IsRegular() {
		img := t.singleImage(imagePath)
		images = []image.Image{img}
	} else {
		message := "path is neither file nor directory"
		t.logger.WithField("imagePath", imagePath).Errorln(message)
		err = errors.New(message)
		return nil, err
	}

	return images, nil
}

func (t *tagger) singleImage(filename string) image.Image {
	return image.New(filename)
}

func (t *tagger) imageBatch(s string) []image.Image {
	// TODO: implement

	return nil
}
