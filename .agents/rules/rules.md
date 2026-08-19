---
trigger: always
always_on: true
---


# Agent Rules & Development Guidelines

This document defines the strict operational rules and conventions for AI Agents working within this repository.

---

## 1. Python Interpreter & Virtual Environment Specifications (CRITICAL)

### 1.1 Virtual Environment Principle
- **Strict Prohibition on Global System Python**: Execution of system global Python or pip (e.g., direct calls like `python`, `python3`, `pip`, `pip3`) is strictly forbidden.
- **Mandatory Virtual Environment Usage**: The Python virtual environment is located at the project parent directory (`/home/b0457812963/Mamba3RL` or `..` relative to workspace root). The AI Agent MUST automatically search for, locate, and execute the Python interpreter and pip binary within this virtual environment.
- **Executable Resolution Order**:
  1. Absolute Path: `/home/b0457812963/Mamba3RL/bin/python`
  2. Relative Path: `../bin/python` or `../bin/pip`
- **Tool / Command Execution**: All script executions, package management commands, and unit test runs MUST explicitly invoke the virtual environment binary (e.g., `../bin/python <script.py>` or `../bin/pip install <package>`).

### 1.2 Development OS Environment
- **Platform**: Ubuntu Linux.
- **Compatibility**: All shell commands, path manipulations, process controls, and environment flags must be formatted and optimized for Ubuntu Linux bash environment.

---

## 2. Language & Documentation Standards

### 2.1 English-Only Code Artifacts
- **Specifications & Specs**: All specifications, design documents, and module specs MUST be written in English.
- **Code Annotations**: All docstrings, inline code comments, type hints, function descriptions, and variable names inside the codebase MUST be written in English.
- **Commit & Log Messages**: Git commit messages, change logs, and technical specification files inside `.agents/` or `spec/` MUST be written in English.

### 2.2 Interactive Communication
- Chat interactions with the user may use Traditional Chinese or the user's preferred language, but all persistent code files and technical documentation added to the codebase MUST strictly follow the English requirement.

---

## 3. Code Integrity & Verification Workflow

### 3.1 Empirical Verification
- **Runtime Testing**: Never mark a task or bug fix as complete without running empirical verification using the virtual environment interpreter (`../bin/python`).
- **No Masking Failures**: Do not resolve errors by ignoring exceptions, introducing empty fallbacks, or disabling failing assertions without root-cause analysis.

### 3.2 Log Inspection & Diagnostics
- Always inspect complete error tracebacks and full log outputs before making diagnostic conclusions or modifying code logic.
