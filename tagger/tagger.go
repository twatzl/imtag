package tagger

import (
	"fmt"
	"github.com/mewkiz/pkg/osutil"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/tagger/image"
	"github.com/twatzl/imtag/tagger/knn"
	"github.com/twatzl/imtag/tagger/label"
	"github.com/twatzl/imtag/tagger/tag"
	"os"
)

type TensorFlowModelType int;

type Tagger interface {
	/**
	 * AddNewLabel will register a new label for the tagger. The label will be stored in
	 * the label storage. Adding a new label does not mean the label is embedded directly, since
	 * the label is only embedded before tagging an image. This way the word2vec implementation can
	 * be changed without the need to embed all the labels again.
	 */
	AddNewLabel(label string) error
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

func (t *tagger) AddNewLabel(label string) error {
	// store synset ids because then we can just swap the embedding algorithm

	// "n" = search for nouns
	synsets := t.conf.WordNet.Search(label)["n"]

	var sid string
	if len(synsets) > 0 {
		sid = synsets[0].Id()
	} else {
		t.logger.WithField("label", label).Warn("word could not be found in WordNet")
		return errors.New(fmt.Sprintf("word %s could not be found in WordNet", label))
	}

	labels, err := t.conf.LabelStorage.loadLabels()
	if err != nil {
		return err
	}
	labels = append(labels, sid)
	err = t.conf.LabelStorage.storeLabels(labels)
	return err
}

func (t *tagger) LoadAndTagImages(imagePath string) (result []image.Image, err error) {
	if t.conf.LabelStorage == nil {
		err = errors.New("no label storage module defined")
		return nil, err
	}

	if t.conf.Word2VecModel == nil {
		err = errors.New("no word2vec model loaded")
		return nil, err
	}

	labels, err := t.conf.LabelStorage.loadLabels()
	if err != nil {
		return nil, err
	}
	embeddedLabels := t.embedKnownLabels(labels)

	images, err := t.loadAndClassifyImages(imagePath)
	if err != nil {
		t.logger.WithError(err).Errorln("error during loading and classification of images")
		return nil, err
	}

	if t.conf.RawClassifierResults {
		return images, nil
	}

	vec := embedImageFlat(t.conf.Word2VecModel, images[0])

	resultLabels := knn.KnnSearch(embeddedLabels, [][]float32{vec}, t.conf.K, knn.CosDist)[0]

	tags := make([]tag.Tag, len(resultLabels))
	for i := range resultLabels {
		// TODO fix confidence
		tags[i] = tag.New(resultLabels[i].GetLabel(), 0)
	}
	images[0].SetTags(tags)

	return
}

func (t* tagger) loadAndClassifyImages(imagePath string) (result []image.Image, err error)  {
	if t.conf.ImageClassifier == nil {
		err = errors.New("no classifier loaded")
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

/**
 * embedKnownLabels embeds the labels which were choosen by the user before in our
 * n-dimensional vector space. This is done on demand so that the user can change the
 * implementation of word2vec without the need to re register all data again.
 */
func (t *tagger) embedKnownLabels(labels []string) (embeddedLabels []label.Label) {

	for _, l := range labels {
		vector := t.conf.Word2VecModel.Word2Vec(l)
		embeddedLabels = append(embeddedLabels, label.New(l, vector))
	}

	return embeddedLabels
}
