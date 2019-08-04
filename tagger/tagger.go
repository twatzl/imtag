package tagger

import (
	"github.com/mewkiz/pkg/osutil"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/tagger/image"
	"github.com/twatzl/imtag/tagger/tag"
	"os"
)

type TensorFlowModelType int;

type Tagger interface {
	EmbedNewLabel(label string) error
	LoadAndTagImages(imagePath string) ([]image.Image, error)
}

type tagger struct {
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

func (t *tagger) EmbedNewLabel(label string) error {
	// TODO: embed the new label somewhere and then save the resulting vector
	panic("implement me")
}

func (t *tagger) LoadAndTagImages(imagePath string) (result []image.Image, err error) {

	if t.conf.ImageClassifier == nil {
		err = errors.New("no classifier loaded")
		return nil, err
	}

	if t.conf.Word2VecModel == nil {
		err = errors.New("no word2vec model loaded")
		return nil, err
	}

	images, err := t.prepareImageBatch(imagePath)
	if err != nil {
		return nil, err
	}

	var tags [][]tag.Tag
	if t.conf.K == 0 {
		tags, err = t.conf.ImageClassifier.ClassifyImages(images)
	} else {
		tags, err = t.conf.ImageClassifier.ClassifyImagesTopK(images, t.conf.K)
	}

	if err != nil {
		err = errors.New("error during image classification")
		return nil, err
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
