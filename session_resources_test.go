// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigma_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/wintermi/sigma"
)

func TestCleanupSessionResourcesPassesSessionID(t *testing.T) {
	calls := make(chan string, 1)
	unregister := sigma.RegisterSessionResourceCleanup(func(sessionID string) error {
		calls <- sessionID
		return nil
	})
	t.Cleanup(unregister)

	if err := sigma.CleanupSessionResources("session-1"); err != nil {
		t.Fatalf("CleanupSessionResources returned error: %v", err)
	}
	if got, want := receiveCleanupCall(t, calls), "session-1"; got != want {
		t.Fatalf("session id = %q, want %q", got, want)
	}
}

func TestCleanupSessionResourcesPassesEmptySessionID(t *testing.T) {
	calls := make(chan string, 1)
	unregister := sigma.RegisterSessionResourceCleanup(func(sessionID string) error {
		calls <- sessionID
		return nil
	})
	t.Cleanup(unregister)

	if err := sigma.CleanupSessionResources(""); err != nil {
		t.Fatalf("CleanupSessionResources returned error: %v", err)
	}
	if got := receiveCleanupCall(t, calls); got != "" {
		t.Fatalf("session id = %q, want empty", got)
	}
}

func TestRegisterSessionResourceCleanupUnregisters(t *testing.T) {
	calls := make(chan string, 1)
	unregister := sigma.RegisterSessionResourceCleanup(func(sessionID string) error {
		calls <- sessionID
		return nil
	})
	unregister()
	unregister()

	if err := sigma.CleanupSessionResources("session-1"); err != nil {
		t.Fatalf("CleanupSessionResources returned error: %v", err)
	}
	select {
	case sessionID := <-calls:
		t.Fatalf("cleanup was called with %q after unregister", sessionID)
	default:
	}
}

func TestCleanupSessionResourcesReturnsJoinedErrorsAndContinues(t *testing.T) {
	firstErr := errors.New("first cleanup failed")
	secondErr := errors.New("second cleanup failed")
	calls := make(chan string, 3)

	unregisterFirst := sigma.RegisterSessionResourceCleanup(func(string) error {
		calls <- "first"
		return firstErr
	})
	t.Cleanup(unregisterFirst)
	unregisterSecond := sigma.RegisterSessionResourceCleanup(func(string) error {
		calls <- "second"
		return secondErr
	})
	t.Cleanup(unregisterSecond)
	unregisterThird := sigma.RegisterSessionResourceCleanup(func(string) error {
		calls <- "third"
		return nil
	})
	t.Cleanup(unregisterThird)

	err := sigma.CleanupSessionResources("session-1")
	if !errors.Is(err, firstErr) {
		t.Fatalf("error = %v, want first error", err)
	}
	if !errors.Is(err, secondErr) {
		t.Fatalf("error = %v, want second error", err)
	}
	got := []string{
		receiveCleanupCall(t, calls),
		receiveCleanupCall(t, calls),
		receiveCleanupCall(t, calls),
	}
	joined := strings.Join(got, ",")
	for _, want := range []string{"first", "second", "third"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("calls = %v, want %q", got, want)
		}
	}
}

func receiveCleanupCall(t *testing.T, calls <-chan string) string {
	t.Helper()

	select {
	case call := <-calls:
		return call
	default:
		t.Fatal("cleanup was not called")
		return ""
	}
}
