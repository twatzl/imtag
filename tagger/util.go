package tagger

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"log"
	"os"
	"runtime"
	"strings"
)

func InitLogger(level log.Level) *log.Logger {
	logger := log.New()
	logger.SetReportCaller(true)
	logger.Formatter = &log.TextFormatter{
		CallerPrettyfier: func(f *runtime.Frame) (string, string) {
			repopath := fmt.Sprintf("%s/src/github.com/twatzl/imtag", os.Getenv("GOPATH"))
			filename := strings.Replace(f.File, repopath, "", -1)
			return fmt.Sprintf("%s()", f.Function), fmt.Sprintf("%s:%d", filename, f.Line)
		},
	}
	logger.SetLevel(level)
	return logger
}
