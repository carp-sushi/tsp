.PHONY: all
all: fmt build

.PHONY: fmt
fmt:
	@gofmt -w main.go

.PHONY: build
build:
	@go build

.PHONY: clean
clean:
	@go clean

.PHONY: run
run:
	@go run main.go
