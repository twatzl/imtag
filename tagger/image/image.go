package image

import "github.com/twatzl/EmbeddingImageTagger/tagger/tag"

type Image interface {
	GetFilename() string
	GetTags() []tag.Tag
	SetTags(tags []tag.Tag)
	AddTag(tag tag.Tag)
}

type image struct {
	filename string
	tags []tag.Tag
}

func (i *image) GetFilename() string {
	return i.filename
}

func (i *image) GetTags() []tag.Tag {
	return i.tags
}

func (i *image) SetTags(tags []tag.Tag) {
	i.tags = tags
}

func (i *image) AddTag(tag tag.Tag) {
	i.tags = append(i.tags, tag)
}

func New(filename string) Image {
	return &image{
		filename: filename,
	}
}