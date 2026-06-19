# Contributing to diart

Thank you for considering contributing to diart! We appreciate your time and effort to help make this project better.

## Before You Start

1. **Search for Existing Issues or Discussions:**
   - Before opening a new issue or discussion, please check if there's already an existing one related to your topic. This helps avoid duplicates and keeps discussions centralized.

2. **Discuss Your Contribution:**
   - If you plan to make a significant change, it's advisable to discuss it in an issue first. This ensures that your contribution aligns with the project's goals and avoids duplicated efforts.

3. **Questions about diart:**
   - For general questions about diart, use the discussion space on GitHub. This helps in fostering a collaborative environment and encourages knowledge-sharing.

## Opening Issues

If you encounter a problem with diart or want to suggest an improvement, please follow these guidelines when opening an issue:

- **Bug Reports:**
  - Clearly describe the error, including any relevant stack traces.
  - Provide a minimal, reproducible example that demonstrates the issue.
  - Mention the version of diart you are using (as well as any dependencies related to the bug).

- **Feature Requests:**
  - Clearly outline the new feature you are proposing.
  - Explain how it would benefit the project.

## Setting Up a Development Environment

diart uses [uv](https://docs.astral.sh/uv/) for dependency management. After [installing uv](https://docs.astral.sh/uv/getting-started/installation/) and the system dependencies listed in the README:

```shell
git clone https://github.com/<your-username>/diart.git
cd diart
uv sync --extra onnx --group dev
uv run pre-commit install
```

You can then run the tests with `uv run pytest` and the linter/formatter with `uv run ruff check .` and `uv run ruff format .`.

## Opening Pull Requests

We welcome and appreciate contributions! To ensure a smooth review process, please follow these guidelines when opening a pull request:

- **Create a Branch:**
  - Work on your changes in a dedicated branch created from `develop`.

- **Commit Messages:**
  - Write clear and concise commit messages, explaining the purpose of each change.

- **Documentation:**
  - Update documentation when introducing new features or making changes that impact existing functionality.

- **Tests:**
  - If applicable, add or update tests to cover your changes.

- **Code Style:**
  - Follow the existing coding style of the project. We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting (it replaces `black` and `isort`).
  - The `pre-commit` hooks run Ruff automatically. You can also run `uv run ruff check --fix .` and `uv run ruff format .` manually.

- **Discuss Before Major Changes:**
  - If your PR includes significant changes, discuss it in an issue first.

- **Follow the existing workflow:**
  - Make sure to open your PR against `develop` (**not** `main`).

## Thank You

Your contributions make diart better for everyone. Thank you for your time and dedication!

Happy coding!
