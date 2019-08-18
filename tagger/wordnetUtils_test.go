package tagger

import (
	"fmt"
	"log"
	"reflect"
	"sort"
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
			[]string{"vertebrate",
				"chordate",
				"living_thing",
				"object",
				"entity",
				"feline",
				"mammal",
				"animal",
				"organism",
				"whole",
				"physical_entity",
				"carnivore",
				"placental"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := loadAncestorLabelsForSynset(tt.args.wn, tt.args.synset);
			// need to sort strings, because map is randomly sorted
			sort.Strings(got)
			sort.Strings(tt.want)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("loadAncestorLabelsForSynset() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_loadAncestorHierarchyForSynset(t *testing.T) {
	// http://dlacombejr.github.io/programming/2015/09/28/visualizing-cifar-10-categories-with-wordnet-and-networkx.html
	// http://dlacombejr.github.io/assets/CIFAR_10-wordnet.png
	wn, err := wordnet.Parse("../data/wordnet/dict")
	if err != nil {
		log.Fatal("cannot load wordnet data")
	}
	//nouns := wn.Search("deer")["n"]
	//fmt.Println(nouns[0].Id())

	type args struct {
		wn     *wordnet.WordNet
		synset *wordnet.Synset
	}
	tests := []struct {
		name string
		args args
		want []LabelHierarchy
	}{
		{
			"TestLoadAncestorHierarchyForSynset_cat",
			args{wn, wn.Synset["n02124272"]},
			[]LabelHierarchy{},
		},
		{
			"TestLoadAncestorHierarchyForSynset_deer",
			args{wn, wn.Synset["n02432691"]},
			[]LabelHierarchy{},
		},
		{
			"TestLoadAncestorHierarchyForSynset_airplane",
			args{wn, wn.Synset["n02694015"]},
			[]LabelHierarchy{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fmt.Println(tt.args.synset)

			got := loadAncestorHierarchyForSynset(tt.args.wn, tt.args.synset);
			// need to sort strings, because map is randomly sorted
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("loadAncestorHierarchyForSynset() = %v, want %v", got, tt.want)
			}
		})
	}
}
