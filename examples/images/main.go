// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/sigmatest"
)

const tinyPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="

func main() {
	if err := describeImageInput(context.Background()); err != nil {
		log.Fatal(err)
	}
	if err := generateImage(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func describeImageInput(ctx context.Context) error {
	model := sigmatest.TextModel()
	model.SupportedInputs = []sigma.ContentBlockType{sigma.ContentBlockText, sigma.ContentBlockImage}

	provider := sigmatest.NewFauxProvider(sigmatest.Script{
		Final: sigma.AssistantMessage{
			Content: []sigma.ContentBlock{sigma.Text("The image fixture is a tiny PNG placeholder.")},
		},
	})
	registry, err := sigmatest.Registry(provider, model)
	if err != nil {
		return err
	}

	client := sigma.NewClient(sigma.WithRegistry(registry))
	final, err := client.Complete(ctx, model, sigma.Request{
		Messages: []sigma.Message{
			sigma.UserContent(
				sigma.Text("Describe this image in one sentence."),
				sigma.ImageBase64("image/png", tinyPNG),
			),
		},
	})
	if err != nil {
		return err
	}
	fmt.Println(assistantText(final))
	return nil
}

func generateImage(ctx context.Context) error {
	provider := sigmatest.NewFauxImageProvider(sigmatest.ImageScript{
		Response: sigma.AssistantImages{
			Images: []sigma.ImageInput{
				sigma.ImageOutputData("image/png", tinyPNG),
			},
		},
	})
	registry, err := sigmatest.ImageRegistry(provider)
	if err != nil {
		return err
	}

	client := sigma.NewClient(sigma.WithRegistry(registry))
	images, err := client.GenerateImages(ctx, sigmatest.ImageModel(), sigma.ImageRequest{
		Prompt:   "A simple blue square icon",
		Size:     string(sigma.ImageSize1024x1024),
		Quality:  string(sigma.ImageQualityLow),
		MIMEType: "image/png",
		Count:    1,
	})
	if err != nil {
		return err
	}
	fmt.Printf("generated %d image(s), first MIME type: %s\n", len(images.Images), images.Images[0].MIMEType)
	return nil
}

func assistantText(final sigma.AssistantMessage) string {
	var out string
	for _, block := range final.Content {
		if block.Type == sigma.ContentBlockText {
			out += block.Text
		}
	}
	return out
}
