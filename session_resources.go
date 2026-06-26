// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigma

import (
	"errors"
	"sync"
)

// SessionResourceCleanup releases cached provider resources for a session.
//
// An empty sessionID asks the cleanup to release all session resources it owns.
type SessionResourceCleanup func(sessionID string) error

var sessionResourceCleanups = struct {
	sync.Mutex
	nextID  uint64
	entries map[uint64]SessionResourceCleanup
}{
	entries: make(map[uint64]SessionResourceCleanup),
}

// RegisterSessionResourceCleanup registers cleanup for provider-owned session
// resources and returns a function that unregisters it.
func RegisterSessionResourceCleanup(cleanup SessionResourceCleanup) func() {
	if cleanup == nil {
		return func() {}
	}

	sessionResourceCleanups.Lock()
	sessionResourceCleanups.nextID++
	id := sessionResourceCleanups.nextID
	sessionResourceCleanups.entries[id] = cleanup
	sessionResourceCleanups.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			sessionResourceCleanups.Lock()
			delete(sessionResourceCleanups.entries, id)
			sessionResourceCleanups.Unlock()
		})
	}
}

// CleanupSessionResources releases registered provider-owned session resources.
// Passing an empty sessionID releases all registered session resources.
func CleanupSessionResources(sessionID string) error {
	cleanups := registeredSessionResourceCleanups()
	errs := make([]error, 0, len(cleanups))
	for _, cleanup := range cleanups {
		if err := cleanup(sessionID); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func registeredSessionResourceCleanups() []SessionResourceCleanup {
	sessionResourceCleanups.Lock()
	defer sessionResourceCleanups.Unlock()

	cleanups := make([]SessionResourceCleanup, 0, len(sessionResourceCleanups.entries))
	for _, cleanup := range sessionResourceCleanups.entries {
		cleanups = append(cleanups, cleanup)
	}
	return cleanups
}
