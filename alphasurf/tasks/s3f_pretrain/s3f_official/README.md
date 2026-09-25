# Upstream S3F surface code

`surface.py` is a verbatim copy of `s3f/surface.py` from
`github.com/DeepGraphLearning/S3F` at commit `2efab6a`.

Refresh it from a clone of that repository:

```
cp <S3F clone>/s3f/surface.py surface.py
```

It is used by `../precompute_s3f_exact.py`, so that the surfaces it builds carry
the same features as the upstream `script/process_surface.py`.
