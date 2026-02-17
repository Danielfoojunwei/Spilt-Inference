.PHONY: test bench proto clean

test:
	python -m pytest tests/ -v

bench:
	python -m benchmarks.bench_split_overhead
	python -m benchmarks.bench_dp_utility

proto:
	python -m grpc_tools.protoc \
		-Isrc/proto \
		--python_out=src/proto \
		--grpc_python_out=src/proto \
		src/proto/split_inference.proto

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
