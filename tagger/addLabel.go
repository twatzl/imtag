package tagger

import "github.com/spf13/cobra"

func AddLabel(cmd *cobra.Command, args []string) {
	tagger := New(cmd, args)
	tagger.EmbedNewLabel()
}