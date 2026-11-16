"""The soundscape-type contract and registry.

A type is a self-contained pipeline (generate, QA, deploy) behind four hooks. Types are
registered by import path and loaded lazily, so a type's heavy dependencies (librosa,
rubberband, ...) are only imported when that type is used.

    class MyType(SoundscapeType):
        name = "mytype"
        summary = "one line for `ss list`"
        def add_gen_args(self, ap): ...
        def gen(self, args, ap): ...        -> exit code; writes build/<type>/<id>/
        def add_deploy_args(self, ap): ...
        def deploy(self, args, ap): ...     -> exit code; writes deploy/<type>/manifest.json
        def list(self): ...                 prints what can be generated
"""

import importlib

REGISTRY = {
    "inducer": "resonance_utils.inducer.type:InducerType",
    "energizer": "resonance_utils.energizer.type:EnergizerType",
}


class SoundscapeType:
    name = ""
    summary = ""

    def add_gen_args(self, ap):
        raise NotImplementedError

    def gen(self, args, ap):
        raise NotImplementedError

    def add_deploy_args(self, ap):
        raise NotImplementedError

    def deploy(self, args, ap):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError


def names():
    return list(REGISTRY)


def get(name):
    if name not in REGISTRY:
        raise KeyError(f"unknown soundscape type {name!r}; choose from {names()}")
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)()
