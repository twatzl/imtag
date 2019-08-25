package label

type Label interface {
	GetLabel() string
	GetVector() []float32
}

func New(l string, vector []float32) Label {
	return &label{
		l,
		vector,
	}
}

type label struct {
	label string
	vector []float32
}

func (e *label) GetLabel() string {
	return e.label
}

func (e *label) GetVector() []float32 {
	return e.vector
}