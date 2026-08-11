# Autonomous Agent

A multi-step agent that works autonomously in an isolated workspace using the Claude Agent SDK built-in file tools (Read, Write, Edit, Glob, Grep), then returns a structured result via `output_format`.

## Run

```
uipath auth
uipath init
uipath run agent '{"input": "Compare the pros and cons of SQLite vs Postgres for a small internal tool and give a recommendation."}'
```

## Debug

```
uipath dev
```
