package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// 1. Setup Base Container
	python := client.Container().From("python:3.12.2-bookworm").
		// Mounts the local 'go/python-files' directory (including code and requirements.txt)
		WithDirectory("/python", client.Host().Directory("python-files")).
		WithWorkdir("/python")

	// 2. INSTALL DEPENDENCIES
	// CRITICAL: Install packages from the mounted requirements.txt file
	python = python.WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	// Optional: Print Python version
	python = python.WithExec([]string{"python", "--version"})

	// 3. EXECUTE the training script
	// This runs only after all modules are installed.
	python = python.WithExec([]string{"python", "train_model.py"})

	// 4. EXPORT the model artifact.
	_, err = python.
		Directory("artifacts").
		Export(ctx, "model")
	if err != nil {
		return err
	}

	return nil
}
