# DataGen: Synthetic Data Generation for LLM Training

DataGen is a Python library for generating high-quality synthetic text datasets specifically designed for training large language models (LLMs). The library enables AI researchers and engineers to generate fully synthetic pretraining datasets, augment existing datasets, and create synthetic fine-tuning data like instruction-response pairs.

## Features

- **Advanced Generation Techniques**: Implements Self-Instruct, Evol-Instruct, and controlled generation methods
- **Quality Control**: Built-in quality filtering and evaluation metrics
- **Privacy Preservation**: Techniques to ensure synthetic data respects privacy constraints
- **Developer-Friendly**: Simple API, CLI tools, and integration with ML pipelines
- **Customizable**: Modular architecture allowing for extension and customization

## Installation

```bash
pip install datagen
```

## Quick Start

```python
from datagen import Generator, Config

# Create a generator with default configuration
generator = Generator(Config())

# Generate synthetic instruction-response pairs
results = generator.generate_from_seed(
    seed_examples=[
        {"instruction": "Explain quantum computing", "response": "Quantum computing uses quantum bits..."},
        {"instruction": "Write a poem about AI", "response": "Silicon dreams and neural streams..."}
    ],
    count=100
)

# Save results to a file
results.save("synthetic_data.jsonl")
```

## Architecture

DataGen consists of several modular components:

1. **Generation Engine**: Creates synthetic text using advanced techniques
2. **Sampling & Configuration**: Manages generation parameters
3. **Quality & Filtering**: Ensures high-quality outputs
4. **Privacy-Preserving**: Protects against sensitive data leakage
5. **Data Pipeline**: Handles input/output and integrates with ML workflows
6. **Developer Tools**: CLI and utilities for easy usage

## Documentation

For detailed documentation, see [docs/](docs/).

## License

MIT License

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. 