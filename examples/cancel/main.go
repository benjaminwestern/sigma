// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/sigmatest"
)

func main() {
	provider := sigmatest.NewFauxProvider(sigmatest.Script{
		WaitForCancel: true,
		Final: sigma.AssistantMessage{
			Content: []sigma.ContentBlock{sigma.Text("partial text before cancellation")},
		},
	})
	registry, err := sigmatest.Registry(provider)
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()

	client := sigma.NewClient(sigma.WithRegistry(registry))
	stream := client.Stream(ctx, sigmatest.TextModel(), sigma.Request{
		Messages: []sigma.Message{sigma.UserText("This request will time out.")},
	})

	final, err := sigma.Collect(context.Background(), stream)
	if err == nil {
		log.Fatal("expected cancellation error")
	}
	if !errors.Is(err, sigma.ErrAborted) {
		log.Fatal(err)
	}

	fmt.Printf("stop reason: %s\n", final.StopReason)
	fmt.Printf("final content blocks: %d\n", len(final.Content))

	var generationErr *sigma.GenerationError
	if errors.As(err, &generationErr) {
		aborted, ok := generationErr.FinalMessage()
		fmt.Printf("aborted final available: %t (%s)\n", ok, aborted.StopReason)
	}
}
