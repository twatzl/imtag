package tensorflowImageClassifier

import (
	"bufio"
	"errors"
	tg "github.com/galeone/tfgo"
	"github.com/sirupsen/logrus"
	log "github.com/sirupsen/logrus"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"github.com/twatzl/EmbeddingImageTagger/tagger/imageClassifier/tensorflowImageClassifier/vggPreprocessing"
	"image"
	"image/draw"
	_ "image/jpeg"
	"io/ioutil"
	"os"
	"path"
)

const graph_tag = "serve"
const input_tag = "input"
const output_tag = "vgg_19/fc8/squeezed"
const labelFile = "imagenet_comp_graph_label_strings.txt"
const VGGModel = "frozen_vgg_19.pb"

const contrib_path = "/usr/local/lib64/python3.7/site-packages/tensorflow_core/contrib/layers"

type TensorFlowClassifier interface {
	LoadSavedModel(modelFolder string, graphTag string)
	LoadFrozenModel(modelFile string) (err error)
	GetTopKPredictionsForImage(imageFilename string, k int) (values [][]float32, indices [][]int32, err error)
	GetLabelsForPredictions(batchValues [][]float32, batchIndices [][]int32) []map[string]float32
}

type tensorFlowClassifier struct {
	dataPath string
	log         *logrus.Logger
	labels      []string
	savedModel  *tg.Model
	frozenModel *tf.Graph
}

func New(dataPath string, log *logrus.Logger) TensorFlowClassifier {
	classifier := &tensorFlowClassifier{
		dataPath: dataPath,
		log: log,
	}

	err := classifier.loadLabels()
	if err != nil {
		log.WithError(err).Errorln("could not load labels")
	}

	return classifier
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
	labeledPredictions := make([]map[string] float32, 0)

	for i := 0 ; i < length; i++ {
		values := batchValues[i]
		indices := batchIndices[i]

		if (len(values) != len(indices)) {
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

func (tfc *tensorFlowClassifier) GetTopKPredictionsForImage(imageFilename string, k int) (values [][]float32, indices [][]int32, err error) {
	inputTensor, err := tfc.loadTensorFromImage(imageFilename)
	if err != nil {
		log.Errorf("Error creating inputTensor tensor: %s\n", err.Error())
		return
	}
	//results, predictions := loadAndRunSavedModel(inputTensor)
	results, predictions, err := tfc.runClassificationModel(inputTensor)
	if err != nil {
		// todo
		log.WithError(err).Errorln("error during run of classifier model")
		return
	}

	log.Debugf("Predictions %v \n", predictions[0])
	valuesTensor, indicesTensor, err := topKPredictions(results[0], k)
	if err != nil {
		log.WithError(err).Errorln("error retrieving top k predictions")
		return
	}
	values = valuesTensor.Value().([][]float32)
	indices = indicesTensor.Value().([][]int32)
	return values, indices, err
}

func (tfc *tensorFlowClassifier) LoadSavedModel(modelFolder string, graphTag string) {
	modelFolder = path.Join(tfc.dataPath, modelFolder)
	log.WithField("folder", modelFolder).Infoln("loading saved model")
	tfc.frozenModel = nil
	tfc.savedModel = tg.LoadModel(modelFolder, []string{graphTag}, nil)
}

func (tfc *tensorFlowClassifier) LoadFrozenModel(modelFile string) (err error) {
	tfc.savedModel = nil
	filepath := path.Join(tfc.dataPath, modelFile)
	log.WithField("file", filepath).Infoln("loading frozen model")
	f, err := os.Open(filepath)
	if err != nil {
		log.WithError(err).Errorln("could not open frozen model file")
		return err
	}
	defer f.Close()

	modelData, err := ioutil.ReadAll(f)
	if err != nil {
		log.WithError(err).Errorln("could not read frozen model file")
		return err
	}

	graph := tf.NewGraph()
	err = graph.Import(modelData, "")
	if err != nil {
		log.WithError(err).Errorln("error decoding frozen model")
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
				tfc.savedModel.Op(output_tag, 0),
			}, map[tf.Output]*tf.Tensor{
				tfc.savedModel.Op(input_tag, 0): imageInputTensor,
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

		input := graph.Operation(input_tag).Output(0)
		output := graph.Operation(output_tag).Output(0)

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

// loadLabels will load the imagenet label from a given textfile (defined in the labelFile constant).
func (tfc *tensorFlowClassifier) loadLabels() (err error) {
		// Load labels
	lfp := path.Join(tfc.dataPath, labelFile)
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
	file, _ := os.Open(name)
	defer file.Close()

	rgbaImg, _, err := tfc.loadImage(name)
	if err != nil {
		log.WithError(err).Errorln("error loading image")
		return nil, err
	}

	/*tensor, err := makeTensorFromImage(rgbaImg)
	if err != nil {

		log.Errorf("Error creating tensor from image: %s \n", err.Error())
		return nil, err
	}*/

	tensor, err := vggPreprocessing.VGGPreprocessingForEval(rgbaImg, 224, 224, 224)
	if err != nil {
		log.WithError(err).Errorln("error during preprocessing of image")
	}

	return tensor, err
}

func (tfc *tensorFlowClassifier) loadImage(filename string) (*image.RGBA, string, error) {
	log.WithField("file", filename).Infoln("loading image")

	file, err := os.Open(filename)
	if err != nil {
		tfc.log.WithField("file", filename).WithError(err).Errorln("could not open file")
		return nil, "", err
	}
	defer file.Close()

	// TODO: check if image is kept in ram
	img, format, err := image.Decode(file)
	if err != nil {
		tfc.log.WithField("file", filename).WithError(err).Errorln("could decode image")
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
