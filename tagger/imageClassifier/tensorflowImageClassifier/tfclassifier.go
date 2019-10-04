package tensorflowImageClassifier

import (
	"bufio"
	"errors"
	tg "github.com/galeone/tfgo"
	"github.com/sirupsen/logrus"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	taggerImage "github.com/twatzl/imtag/tagger/image"
	"github.com/twatzl/imtag/tagger/imageClassifier"
	"github.com/twatzl/imtag/tagger/imageClassifier/tensorflowImageClassifier/vggPreprocessing"
	"github.com/twatzl/imtag/tagger/tag"
	"image"
	"image/draw"
	_ "image/jpeg"
	"io/ioutil"
	"os"
)

type TensorFlowClassifier interface {
	LoadSavedModel(modelFolder string, graphTag string)
	LoadFrozenModel(modelFile string) (err error)
	GetTopKPredictionsForImage(imageFilename string, k int) (values [][]float32, indices [][]int32, err error)
	GetLabelsForPredictions(batchValues [][]float32, batchIndices [][]int32) []map[string]float32
	imageClassifier.ImageClassifier
}

type tensorFlowClassifier struct {
	dataPathForResource func(string) string
	log                 *logrus.Logger
	labels              []string
	savedModel          *tg.Model
	frozenModel         *tf.Graph
	config              Config
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
}

func New(dataPathMapper func(string) string, log *logrus.Logger, config Config) TensorFlowClassifier {
	classifier := &tensorFlowClassifier{
		dataPathForResource: dataPathMapper,
		log:                 log,
		config:              config,
	}

	err := classifier.loadLabels()
	if err != nil {
		log.WithError(err).Errorln("could not load labels")
	}

	return classifier
}

func (tfc *tensorFlowClassifier) ClassifyImages(images []taggerImage.Image) ([][]tag.Tag, error) {
	tags := [][]tag.Tag{}
	for _, img := range images {

		// TODO: tensorflow supports batch processing. increase performance by using that
		values, err := tfc.GetPredictionsForImage(img.GetFilename())
		if err != nil {
			return tags, err
		}

		seq := make([]int32, tfc.config.NumLabels)
		for i := range seq {
			seq[i] = int32(i)
		}

		labeledResults := tfc.GetLabelsForPredictions(values, [][]int32{seq})[0]

		tagsForImage := []tag.Tag{}
		for key, val := range labeledResults {
			tagsForImage = append(tagsForImage, tag.New(key, val))
		}

		tags = append(tags, tagsForImage)
	}
	return tags, nil
}

func (tfc *tensorFlowClassifier) ClassifyImagesTopK(images []taggerImage.Image, k int) ([][]tag.Tag, error) {
	tags := [][]tag.Tag{}
	for _, img := range images {

		// TODO: tensorflow supports batch processing. increase performance by using that
		values, indices, err := tfc.GetTopKPredictionsForImage(img.GetFilename(), k)
		if err != nil {
			return tags, err
		}

		labeledResults := tfc.GetLabelsForPredictions(values, indices)[0]

		tagsForImage := []tag.Tag{}
		for key, val := range labeledResults {
			tagsForImage = append(tagsForImage, tag.New(key, val))
		}

		tags = append(tags, tagsForImage)

	}
	return tags, nil
}

// GetLabelsForPredictions returns a map of labels to values for each element in a batch. Usually this function is called
// after the top k predictions have been taking from the classification.
func (tfc *tensorFlowClassifier) GetLabelsForPredictions(batchValues [][]float32, batchIndices [][]int32) []map[string]float32 {
	/*for i := 0; i < len(values[0]); i++ {
		fmt.Printf("%f \t %s\n", values[0][i], tfc.labels[indices[0][i]])
	}*/

	if len(batchValues) != len(batchIndices) {
		tfc.log.Errorln("batch size for values and indices must match")
		return nil
	}

	length := len(batchValues)
	labeledPredictions := make([]map[string]float32, 0)

	for i := 0; i < length; i++ {
		values := batchValues[i]
		indices := batchIndices[i]

		if len(values) != len(indices) {
			tfc.log.WithField("batchIndex", i).Errorln("number of values and labels must match")
			continue;
		}

		mapping := make(map[string]float32)

		for j := 0; j < len(values); j++ {
			label := tfc.labels[indices[j]]
			value := values[j]
			mapping[label] = value
		}

		labeledPredictions = append(labeledPredictions, mapping)
	}

	return labeledPredictions
}

func (tfc *tensorFlowClassifier) GetPredictionsForImage(imageFilename string) (values [][]float32, err error) {
	inputTensor, err := tfc.loadTensorFromImage(imageFilename)
	if err != nil {
		tfc.log.Errorf("Error creating inputTensor tensor: %s\n", err.Error())
		return
	}
	_, predictions, err := tfc.runClassificationModel(inputTensor)
	if err != nil {
		// todo
		tfc.log.WithError(err).Errorln("error during run of classifier model")
		return
	}
	return predictions, err
}

func (tfc *tensorFlowClassifier) GetTopKPredictionsForImage(imageFilename string, k int) (values [][]float32, indices [][]int32, err error) {
	inputTensor, err := tfc.loadTensorFromImage(imageFilename)
	if err != nil {
		tfc.log.Errorf("Error creating inputTensor tensor: %s\n", err.Error())
		return
	}
	results, predictions, err := tfc.runClassificationModel(inputTensor)
	if err != nil {
		// todo
		tfc.log.WithError(err).Errorln("error during run of classifier model")
		return
	}

	tfc.log.Debugf("Predictions %v \n", predictions[0])
	valuesTensor, indicesTensor, err := topKPredictions(results[0], k)
	if err != nil {
		tfc.log.WithError(err).Errorln("error retrieving top k predictions")
		return
	}
	values = valuesTensor.Value().([][]float32)
	indices = indicesTensor.Value().([][]int32)
	return values, indices, err
}

func (tfc *tensorFlowClassifier) LoadSavedModel(modelFolder string, graphTag string) {
	modelFolder = tfc.dataPathForResource(modelFolder)
	tfc.log.WithField("folder", modelFolder).Infoln("loading saved model")
	tfc.frozenModel = nil
	tfc.savedModel = tg.LoadModel(modelFolder, []string{graphTag}, nil)
}

func (tfc *tensorFlowClassifier) LoadFrozenModel(modelFile string) (err error) {
	tfc.savedModel = nil
	filepath := tfc.dataPathForResource(modelFile)
	tfc.log.WithField("file", filepath).Infoln("loading frozen model")
	f, err := os.Open(filepath)
	if err != nil {
		tfc.log.WithError(err).Errorln("could not open frozen model file")
		return err
	}
	defer f.Close()

	modelData, err := ioutil.ReadAll(f)
	if err != nil {
		tfc.log.WithError(err).Errorln("could not read frozen model file")
		return err
	}

	graph := tf.NewGraph()
	err = graph.Import(modelData, "")
	if err != nil {
		tfc.log.WithError(err).Errorln("error decoding frozen model")
		return err
	}

	tfc.frozenModel = graph
	return nil
}

// runClassificationModel will run a previously loaded model on a given input tensor
// and returns the raw results as well as the predictions from the classification.
func (tfc *tensorFlowClassifier) runClassificationModel(imageInputTensor *tf.Tensor) (
	results []*tf.Tensor,
	predictions [][]float32,
	err error) {

	/** SavedModel **/
	if tfc.savedModel != nil {
		results = tfc.savedModel.Exec(
			[]tf.Output{
				tfc.savedModel.Op(tfc.config.OutputTag, 0),
			}, map[tf.Output]*tf.Tensor{
				tfc.savedModel.Op(tfc.config.InputTag, 0): imageInputTensor,
			},
		)
		predictions = results[0].Value().([][]float32)
		/** Frozen graph **/
	} else if tfc.frozenModel != nil {
		graph := tfc.frozenModel

		session, err := tf.NewSession(graph, nil)
		if err != nil {
			return nil, nil, err
		}
		defer session.Close()

		input := graph.Operation(tfc.config.InputTag).Output(0)
		output := graph.Operation(tfc.config.OutputTag).Output(0)

		results, err = session.Run(
			map[tf.Output]*tf.Tensor{input: imageInputTensor},
			[]tf.Output{output},
			nil)
		if err != nil {
			return nil, nil, err
		}

		predictions = results[0].Value().([][]float32)
	} else {
		err := errors.New("model must be loaded before starting classification")
		tfc.log.Errorln(err.Error())
		return nil, nil, err
	}

	return results, predictions, nil
}

// loadLabels will load the imagenet label from a given textfile (defined in the LabelFile constant).
func (tfc *tensorFlowClassifier) loadLabels() (err error) {
	// Load labels
	lfp := tfc.dataPathForResource(tfc.config.LabelFile)
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

	tfc.labels = labels
	return nil
}

func (tfc *tensorFlowClassifier) loadTensorFromImage(name string) (*tf.Tensor, error) {
	rgbaImg, _, err := tfc.loadRGBAImage(name)
	if err != nil {
		tfc.log.WithError(err).Errorln("error loading image")
		return nil, err
	}

	/*tensor, err := makeTensorFromImage(rgbaImg)
	if err != nil {

		log.Errorf("Error creating tensor from image: %s \n", err.Error())
		return nil, err
	}*/

	tensor, err := vggPreprocessing.VGGPreprocessingForEval(rgbaImg, 224, 224, 224)
	if err != nil {
			tfc.log.WithError(err).Errorln("error during preprocessing of image")
	}

	return tensor, err
}

func (tfc *tensorFlowClassifier) loadRGBAImage(filename string) (rgbaImage *image.RGBA, format string, err error) {
	tfc.log.WithField("file", filename).Infoln("loading image")

	file, err := os.Open(filename)
	if err != nil {
		tfc.log.WithField("file", filename).WithError(err).Errorln("could not open file")
		return nil, "", err
	}
	defer file.Close()

	// TODO: check if image is kept in ram
	img, format, err := image.Decode(file)
	if err != nil {
		tfc.log.WithField("file", filename).WithError(err).Errorln("could not decode image")
		return nil, "", err
	}

	rect := img.Bounds()
	rgbaImg := image.NewRGBA(rect)
	draw.Draw(rgbaImg, rect, img, rect.Min, draw.Src)

	return rgbaImg, format, nil
}

func topKPredictions(predictions *tf.Tensor, k int) (val, idx *tf.Tensor, err error) {
	graph, input, values, indices, err := makeTopKGraph(k)
	if err != nil {
		return nil, nil, err
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, nil, err
	}
	defer session.Close()
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{input: predictions},
		[]tf.Output{values, indices},
		nil)
	if err != nil {
		return nil, nil, err
	}
	return output[0], output[1], err

}

func makeTopKGraph(k int) (graph *tf.Graph, input, values, indices tf.Output, err error) {
	s := op.NewScope()

	input = op.Placeholder(s, tf.Float)
	values, indices = op.TopKV2(s, input, tg.Const(s, int32(k)))
	graph, err = s.Finalize()

	return graph, input, values, indices, err
}

// based on https://outcrawl.com/image-recognition-api-go-tensorflow
func makeTensorFromImage(image *image.RGBA) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(image.Pix)
	if err != nil {
		return nil, err
	}
	graph, input, output, err := makeTransformImageGraph(&image.Rect)
	if err != nil {
		return nil, err
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func makeTransformImageGraph(rect *image.Rectangle) (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W  = 224, 224
		Mean  = float32(117) // was 117
		Scale = float32(1)
	)
	s := op.NewScope()
	//TODO: refactor
	input = op.Placeholder(s, tf.Uint8)

	height := rect.Max.Y - rect.Min.Y
	width := rect.Max.X - rect.Min.X

	img := op.Reshape(s, op.Cast(s, input, tf.Float), tg.Const(s, [3]int32{int32(height), int32(width), 4}))
	// remove alpha channel
	img = op.Slice(s, img, tg.Const(s, [3]int32{0, 0, 0}), tg.Const(s, [3]int32{int32(height), int32(width), 3}))
	// Create a batch containing a single image
	batch := op.ExpandDims(s,
		// Use pixel values
		img,
		op.Const(s.SubScope("make_batch"), int32(0)))

	// Resize to 224x224 with bilinear interpolation
	resizedBatch := op.ResizeBilinear(s,
		batch,
		op.Const(s.SubScope("size"), []int32{H, W}))

	// Div and Sub perform (value-Mean)/Scale for each pixel
	output = op.Div(s,
		op.Sub(s,
			resizedBatch,
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))

	graph, err = s.Finalize()
	return graph, input, output, err
}

func dummyInputBatch(numImages int, height int, width int, channels int) (*tf.Tensor, error) {
	batch := make([][][][]float32, numImages)
	for i, _ := range batch {
		cols := make([][][]float32, height)
		for x := 0; x < width; x++ {
			rows := make([][]float32, width)
			for y := 0; y < height; y++ {
				pixel := make([]float32, channels)
				rows[y] = pixel
			}
			cols[x] = rows
		}
		batch[i] = cols
	}

	return tf.NewTensor(batch)
}

func dummyInputTensor(size int) (*tf.Tensor, error) {

	imageData := [][]float32{make([]float32, size)}
	return tf.NewTensor(imageData)
}
