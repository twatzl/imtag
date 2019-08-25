package tagger

type embeddedLabelStorage interface {
	storeLabels(labels []string) (err error)
	loadLabels() (labels []string, err error)
}
