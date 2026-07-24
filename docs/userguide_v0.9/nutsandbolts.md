% module xftsim

# Nuts and bolts

These two pages cover the low-level data structures used throughout
`xftsim`. They were a prerequisite for the legacy user guide; in v0.9
they're more optional because the formula DSL hides most of the
indexing concerns, but they're still useful when you want to drop down
to a numpy view or inspect what the simulator is actually carrying
around between generations.

```{toctree}
:maxdepth: 4

Indexing <indexing>
Data structures <struct>
```
