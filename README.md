# Moudular RAG

## Project Overview

This repository contains the implementation of the Modular Pipeline for RAG. The project focuses on modularizing RAG pipeline, making chatbot interface for QA, deploying the application using Docker and evaluation of RAG pipeline using both deterministic and non-deterministic methods.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Features

- Modularized RAG pipeline
- Sphinx documentation
- Application deployment
- Containerization with Docker
- Docker image publishing
- Discord bot integration
- Research paper submission

## Installation

1. Clone the repository:
    ```bash
    git clone -b main https://github.com/eYSIP-2024/26_Assessment_RAG.git
    cd 26_Assessment_RAG
    ```

2. Build the Docker image:
    ```bash
    docker build -t eysip .
    
3. Outside the Docker container, start a tmux session:
    ```bash
    tmux
    ```

4. Run the Docker container with GPU support:
    ```bash
    docker run --rm -it --gpus=all -p 8000:8000 -p 8001:8001 eysip:latest
    ```
5. In the tmux session, navigate to the `inference` directory and run the setup script:
    ```bash
    cd inference/
    chmod +x script.sh
    ./script.sh
    ```

6. Detach from the tmux session by pressing `Ctrl+B` followed by `D`.

## Usage

To start using the application, follow the installation steps above. The application will be available at `http://<serverip>:8000`.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [Chainlit](https://github.com/chainlit/chainlit)
