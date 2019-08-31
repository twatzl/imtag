package tagger

import (
	"github.com/sirupsen/logrus"
	"io/ioutil"
	"os"
	"strings"
)

/**
 * LabelStorage will provide means to store and load labels.
 */
type LabelStorage interface {
	StoreLabels(labels map[string]interface{}) (err error)
	LoadLabelsMap() (labels map[string]interface{}, err error)
	LoadLabelsSlice() (labels []string, err error)
}

type FileLabelStorage interface {
	LabelStorage
	ReadFile() error
	WriteFile() error
}

type fileLabelStorage struct {
	labels map[string]interface{}
	logger *logrus.Logger
	path   string
}

func NewFileLabelStorage(logger *logrus.Logger, path string) FileLabelStorage {
	return &fileLabelStorage{
		nil,
		logger,
		path,
	}
}

func (f *fileLabelStorage) StoreLabels(labels map[string]interface{}) error {
	f.labels = labels
	return nil
}

func (f *fileLabelStorage) LoadLabelsMap() (labels map[string]interface{}, err error) {
	return f.labels, nil
}

func (f *fileLabelStorage) LoadLabelsSlice() (labels []string, err error) {
	return mapToSlice(f.labels), nil
}

func (f *fileLabelStorage) ReadFile() error {
	f.logger.Info("loading label store")
	data, err := ioutil.ReadFile(f.path)
	if os.IsNotExist(err) {
		f.logger.Info("label store does not exist, creating new label store")
		f.labels = make(map[string]interface{})
		return nil
	}
	if err != nil {
		return err
	}

	labels := strings.Split(string(data), "\n")
	f.labels = sliceToMap(labels)
	f.logger.Info("label store loaded")
	return nil
}

func (f *fileLabelStorage) WriteFile() error {
	if f.labels == nil {
		return nil
	}

	f.logger.Info("writing label store")
	data := strings.Join(mapToSlice(f.labels), "\n")
	err := os.Remove(f.path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	if err != nil && os.IsNotExist(err) {
		f.logger.Info("label store does not exist, writing new label store file")
	}

	err = ioutil.WriteFile(f.path, []byte(data), 0755)
	f.logger.Info("label store written")
	return err
}

func mapToSlice(m map[string]interface{}) []string {
	s := []string{}
	for key, _ := range m {
		s = append(s, key)
	}
	return s
}

func sliceToMap(l []string) map[string]interface{} {
	m := make(map[string]interface{})
	for _, val := range l {
		m[val] = ""
	}
	return m
}
