package gocvTfClassifier

import (
	"bufio"
	"fmt"
	"github.com/sirupsen/logrus"
	"github.com/twatzl/imtag/tagger/image"
	"github.com/twatzl/imtag/tagger/imageClassifier"
	"github.com/twatzl/imtag/tagger/tag"
	"gocv.io/x/gocv"
	goimage "image"
	"os"
)



/*
 * gocvTensorFlowClassifier loads a tensorflow model and uses that as classifier. For loading and executing it uses OpenCV
 */
type GoCVTensorFlowClassifier interface {
	imageClassifier.ImageClassifier
	LoadFrozenModel(modelFile string) (err error)
}

type gocvTensorFlowClassifier struct {
	dataPathForResource func(string) string
	log                 *logrus.Logger
	labels              []string
	config              Config
	net                 gocv.Net
	modelPath           string
}

type Config struct {
	/* in newer tensorflow file format the graph has to have a tag */
	GraphTag string // = "serve"
	/* the name with which the input node of a graph is labelled */
	InputTag string //= "input"
	/* the name with which the output node of a graph is labelled */
	OutputTag string // = "resnet_v2_152/predictions/Reshape_1"
	/* the name of the file which contains the imagenet label mappings */
	LabelFile string // = "imagenet_comp_graph_label_strings.txt"
	/* the number of labels (or classes) this needs to be configurable since some models use 1000 and some 1001 */
	NumLabels int

	Backend gocv.NetBackendType
	Target gocv.NetTargetType
}

func New(dataPathMapper func(string) string, log *logrus.Logger, config Config) GoCVTensorFlowClassifier {
	classifier := &gocvTensorFlowClassifier{
		dataPathForResource: dataPathMapper,
		log:                 log,
		labels:              nil,
		net:                 gocv.Net{},
		config:              config,
	}

	err := classifier.loadLabels()
	if err != nil {
		log.WithError(err).Errorln("could not load labels")
	}

	return classifier
}

// loadLabels will load the imagenet label from a given textfile (defined in the LabelFile constant).
func (gcvc *gocvTensorFlowClassifier) loadLabels() (err error) {
	// Load labels
	lfp := gcvc.dataPathForResource(gcvc.config.LabelFile)
	labelsFile, err := os.Open(lfp)
	if err != nil {
		return err
	}
	defer labelsFile.Close()

	scanner := bufio.NewScanner(labelsFile)
	// Labels are separated by newlines
	labels := []string{}
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	gcvc.labels = labels
	return nil
}

func (gcvc *gocvTensorFlowClassifier) LoadFrozenModel(modelFile string) (err error) {
	filepath := gcvc.dataPathForResource(modelFile)
	gcvc.modelPath = filepath
	/*gcvc.log.WithField("file", filepath).Infoln("loading frozen model")
	gcvc.net = gocv.ReadNet(filepath, "")
	err = gcvc.net.SetPreferableBackend(gcvc.config.Backend)
	if err != nil {
		gcvc.log.WithError(err).Errorln("error when setting backend for tensorflow network")
		return err
	}
	err = gcvc.net.SetPreferableTarget(gcvc.config.Target)
	if err != nil {
		gcvc.log.WithError(err).Errorln("error when setting targetl for tensorflow network")
		return err
	}*/

	return nil
}

func (gcvc *gocvTensorFlowClassifier) ClassifyImages(image []image.Image) ([][]tag.Tag, error) {
	/*if gcvc.net.Empty() {
		return nil, errors.New("no gocv net loaded")
	}*/

	gcvc.log.WithField("file", gcvc.modelPath).Infoln("loading frozen model")
	net := gocv.ReadNet(gcvc.modelPath, "")

	for _, curImage := range image {
		filename := curImage.GetFilename()

		// read image from file
		img := gocv.IMRead(filename, gocv.IMReadColor)
		if img.Empty() {
			gcvc.log.Warn("error reading image")
		}

		// convert to a 224x244 image blob that can be processed by Tensorflow
		blob := gocv.BlobFromImage(img, 1.0, goimage.Pt(224, 244), gocv.NewScalar(0, 0, 0, 0), true, false)
		//defer blob.Close()

		// feed the blob into the classifier
		net.SetInput(blob, gcvc.config.InputTag)

		// run a forward pass thru the network
		prob := net.Forward(gcvc.config.OutputTag)
		//defer prob.Close()

		// reshape the results into a 1x1000 matrix
		probMat := prob.Reshape(1, 1)
		//defer probMat.Close()

		// determine the most probable classification, and display it
		_, maxVal, _, maxLoc := gocv.MinMaxLoc(probMat)
		fmt.Printf("maxLoc: %v, maxVal: %v\n", maxLoc, maxVal)

		gocv.WaitKey(1)
	}

	return nil, nil
}

func (gcvc *gocvTensorFlowClassifier) ClassifyImagesTopK(image []image.Image, k int) ([][]tag.Tag, error) {
	return gcvc.ClassifyImages(image)
}
