# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detector import build
from .detector import build_dynamic
from .detector import build_svit


def build_model(args):
    return build(args)

def build_dynamic_model(args):
    return build_dynamic(args)

def build_svit_model(args):
    return build_svit(args)
