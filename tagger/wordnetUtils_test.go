package tagger

import (
	"fmt"
	"log"
	"reflect"
	"testing"

	"github.com/fluhus/gostuff/nlp/wordnet"
)

func Test_loadAncestorLabelsForSynset(t *testing.T) {
	wn, err := wordnet.Parse("../data/wordnet/dict")
	if err != nil {
		log.Fatal("cannot load wordnet data")
	}
	//catNouns := wn.Search("mouse")["n"]
	ss := wn.Synset["n02124272"]
	fmt.Println(ss.String())


	type args struct {
		wn     *wordnet.WordNet
		synset *wordnet.Synset
	}
	tests := []struct {
		name string
		args args
		want []string
	}{
		{
			"TestLoadAncestorLabelsForSynset",
			args{wn, ss},
			[]string{"feline",
				"carnivore",
				"placental",
				"mammal",
				"vertebrate",
				"chordate",
				"animal",
				"organism",
				"living_thing",
				"whole",
				"object",
				"physical_entity",
				"entity"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := loadAncestorLabelsForSynset(tt.args.wn, tt.args.synset); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("loadAncestorLabelsForSynset() = %v, want %v", got, tt.want)
			}
		})
	}
}
