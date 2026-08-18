# seal

## answer style

be brief, use simple language.

## code guidelines

1. use shadcn/ui (https://ui.shadcn.com/docs/components.md) for all ui features. keep things stock as much as possible and avoid custom components.
2. in python, import by module (unless it's `typing`) to improve namespacing and make it easier to navigate code.
3. minimize the number of helper functions, prioritize locality of behavior.
4. keep apis as small as possible. keep public apis even smaller, try to shrink them to one function / object.
5. test file structure should mirror app's file structure, e.g. `agent/proto.py` -> `tests/agent/test_proto.py`. this helps project navigation a lot.

## project setup

1. use uv to manage python
2. use pnpm to manage typescript

## references

ai-python: .reference/ai-python
python sdk (workflow, sandbox, etc.): .reference/vercel-py
