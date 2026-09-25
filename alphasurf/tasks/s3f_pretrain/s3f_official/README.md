# Upstream S3F surface code

`surface.py` is a verbatim copy of `s3f/surface.py` from
`github.com/DeepGraphLearning/S3F` at commit `2efab6a`, vendored here so that
`../precompute_s3f_exact.py` builds surfaces with the same features as the
upstream `script/process_surface.py`. Nothing needs to be downloaded: the copy
in this directory is the one that runs.

To update it against a newer upstream, copy the file over from a checkout of
that repository:

```
cp <S3F checkout>/s3f/surface.py surface.py
```
