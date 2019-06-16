package skipGramModel

import (
	"reflect"
	"testing"
)

const base_path = "../../models/flickr4m.skipGramModel/skipGramModel/tagvec500"


func Test_skipGramModel_integration(t *testing.T) {
	w := New(base_path)
	vec := w.Word2Vec("test")
	println(vec)
}

func Test_skipGramModel_Word2Vec(t *testing.T) {
	type fields struct {
		numImages     int
		numDims       int64
		names         []string
		shapeFilePath string
		idFilePath    string
		dataFilePath  string
		name2Index    map[string]int
	}
	type args struct {
		word string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   []float32
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &skipGramModel{
				numImages:     tt.fields.numImages,
				numDims:       tt.fields.numDims,
				names:         tt.fields.names,
				shapeFilePath: tt.fields.shapeFilePath,
				idFilePath:    tt.fields.idFilePath,
				dataFilePath:  tt.fields.dataFilePath,
				name2Index:    tt.fields.name2Index,
			}
			if got := w.Word2Vec(tt.args.word); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("skipGramModel.Word2Vec() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_skipGramModel_GetDim(t *testing.T) {
	type fields struct {
		numImages     int
		numDims       int64
		names         []string
		shapeFilePath string
		idFilePath    string
		dataFilePath  string
		name2Index    map[string]int
	}
	tests := []struct {
		name   string
		fields fields
		want   int
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &skipGramModel{
				numImages:     tt.fields.numImages,
				numDims:       tt.fields.numDims,
				names:         tt.fields.names,
				shapeFilePath: tt.fields.shapeFilePath,
				idFilePath:    tt.fields.idFilePath,
				dataFilePath:  tt.fields.dataFilePath,
				name2Index:    tt.fields.name2Index,
			}
			if got := w.GetDim(); got != tt.want {
				t.Errorf("skipGramModel.GetDim() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNew(t *testing.T) {
	type args struct {
		basePath string
	}
	tests := []struct {
		name string
		args args
		want skipGramModel
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := New(tt.args.basePath); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_check(t *testing.T) {
	type args struct {
		err error
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			check(tt.args.err)
		})
	}
}

func Test_word2vec_read(t *testing.T) {
	type fields struct {
		numImages     int
		numDims       int64
		names         []string
		shapeFilePath string
		idFilePath    string
		dataFilePath  string
		name2Index    map[string]int
	}
	type args struct {
		word string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   []float32
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &skipGramModel{
				numImages:     tt.fields.numImages,
				numDims:       tt.fields.numDims,
				names:         tt.fields.names,
				shapeFilePath: tt.fields.shapeFilePath,
				idFilePath:    tt.fields.idFilePath,
				dataFilePath:  tt.fields.dataFilePath,
				name2Index:    tt.fields.name2Index,
			}
			if got := w.read(tt.args.word); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("skipGramModel.read() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFloat32frombytes(t *testing.T) {
	type args struct {
		bytes []byte
	}
	tests := []struct {
		name string
		args args
		want float32
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Float32frombytes(tt.args.bytes); got != tt.want {
				t.Errorf("Float32frombytes() = %v, want %v", got, tt.want)
			}
		})
	}
}
