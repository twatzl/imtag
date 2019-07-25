package tag

type Tag interface {
	GetLabel() string
	GetConfidence() float32
}

type tag struct {
	label string
	confidence float32
}

func (t *tag) GetLabel() string {
	return t.label
}

func (t *tag) GetConfidence() float32 {
	return t.confidence
}

func New(label string, confidence float32) Tag {
	return &tag{
		label: label,
		confidence: confidence,
	}
}
