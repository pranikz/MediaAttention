"""UNISAL (Droste, Jiao & Noble, ECCV 2020), vendored from https://github.com/rdroste/unisal
at commit 7f8ca1a7ecb3dbc1bf7fca837cb6186a85ac27d9 under the Apache License 2.0 (see LICENSE in this folder).

Only the model code is vendored (model.py, utils.py, models/). Changes: model.py no longer
prints the torch device at import. Weights are downloaded at runtime (see saliency.unisal).
"""
