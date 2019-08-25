package skipGramModel

import (
	"github.com/twatzl/imtag/tagger/word2vec"
	"encoding/binary"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path"
	"strconv"
	"strings"
)

const shapeFile = "shape.txt"
const idFile = "id.txt"
const dataFile = "feature.bin"

const shapefile_num_words_index = 0
const shapefile_dim_index = 1

type skipGramModel struct {
	numWords      int
	numDims       int64
	names         []string
	shapeFilePath string
	idFilePath    string
	dataFilePath  string
	name2Index    map[string]int
}

func (w *skipGramModel) Word2Vec(word string) []float32 {
	return w.read(word)
}

func (w *skipGramModel) GetDim() int {
	return int(w.numDims)
}

func New(basePath string) word2vec.Word2Vec {
	w := &skipGramModel{}
	w.init(basePath)
	return w
}

func (w *skipGramModel) init(basePath string) {
	w.shapeFilePath = path.Join(basePath, shapeFile)
	w.idFilePath = path.Join(basePath, idFile)
	w.dataFilePath = path.Join(basePath, dataFile)

	// load shape
	filedata, err := ioutil.ReadFile(w.shapeFilePath)
	check(err)
	dataStr := strings.TrimSpace(string(filedata))
	data := strings.Split(dataStr, " ")

	numWords, err := strconv.ParseInt(data[shapefile_num_words_index], 10, 32)
	check(err)
	w.numWords = int(numWords)
	numDims, err := strconv.ParseInt(data[shapefile_dim_index], 10, 32)
	check(err)
	w.numDims = numDims

	// load labels
	filedata, err = ioutil.ReadFile(w.idFilePath)
	check(err)
	w.names = strings.Split(string(filedata), " ")

	// indexing
	w.name2Index = make(map[string]int, len(w.names))
	for i, word := range w.names {
		w.name2Index[word] = i
	}
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func (w *skipGramModel) read(word string) []float32 {
	const FLOAT32_SIZE = 4
	wordIndex := w.name2Index[word]
	offset := w.numDims * int64(wordIndex) * FLOAT32_SIZE

	// open file
	f, err := os.Open(w.dataFilePath)
	check(err)
	defer f.Close()

	_, err = f.Seek(offset, 0)
	check(err)

	rawdata := make([]byte, w.numDims*FLOAT32_SIZE)
	n, err := f.Read(rawdata)
	check(err)
	log.Printf("%d bytes read from file\n", n)

	data := make([]float32, w.numDims)

	for i := 0; i < len(data); i++ {
		conv := rawdata[i*FLOAT32_SIZE : i*FLOAT32_SIZE+FLOAT32_SIZE]
		data[i] = Float32frombytes(conv)
	}

	return data
}

func Float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

