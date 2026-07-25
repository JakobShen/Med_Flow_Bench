# Midscene QuPath Sidecar

This package is intentionally isolated from the Python host.

The Python code only expects a tiny CLI contract:

- `env-check`
- `act`
- `query`
- `assert`

Install it wherever you want, then point the Python host at the folder with:

```bash
export MIDSCENE_QUPATH_SIDECAR_DIR="/absolute/path/to/repository/pathology/runtime/qupath_sidecar"
```

Install dependencies:

```bash
npm install
```

Quick environment check:

```bash
npx --no-install tsx src/cli.ts env-check
```

This v1 sidecar uses the official `@midscene/computer` backend and constrains it to QuPath by:

- focusing QuPath deterministically before each call,
- injecting QuPath-only context into Midscene,
- exposing only app-scoped CLI entry points to the Python host.

That keeps the boundary replaceable. A future stricter custom-interface backend can implement the same CLI without changing the Python host.
