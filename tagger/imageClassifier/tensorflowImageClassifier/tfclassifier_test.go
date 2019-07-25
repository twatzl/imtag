package tensorflowImageClassifier

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"os"
	"runtime"
	"strings"
	"testing"
)

func TestTensorFlowClassifier_GetTopKPredictionsForImage_VGGModel(t *testing.T) {
	logger := log.New()
	logger.SetReportCaller(true)
	logger.Formatter = &log.TextFormatter{
		CallerPrettyfier: func(f *runtime.Frame) (string, string) {
			repopath := fmt.Sprintf("%s/src/github.com/twatzl/imtag", os.Getenv("GOPATH"))
			filename := strings.Replace(f.File, repopath, "", -1)
			return fmt.Sprintf("%s()", f.Function), fmt.Sprintf("%s:%d", filename, f.Line)
		},
	}
	logger.SetLevel(log.DebugLevel)

	imageFilename := "../../../data/pics/Bus-en.jpg"
	k := 10

	tfc := New("../../../data/tensorflowModels/", logger)

	err := tfc.LoadFrozenModel(VGGModel)
	if err != nil {
		t.Fail()
	}

	values, indices, err := tfc.GetTopKPredictionsForImage(imageFilename, k)
	if err !=nil {
		t.Fail()
	}
	labelMapping := tfc.GetLabelsForPredictions(values, indices)
	log.WithField("labelMapping", labelMapping[0]).Info("result")
}