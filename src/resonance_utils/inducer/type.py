from ..core.types import SoundscapeType
from . import deploy as _deploy
from . import gen as _gen


class InducerType(SoundscapeType):
    name = "inducer"
    summary = "power-nap inducer: seamless texture loops with a baked 0.1-0.25 Hz swell (+ tone layer)"

    def add_gen_args(self, ap):
        _gen.add_args(ap)

    def gen(self, args, ap):
        return _gen.run(args, ap)

    def add_deploy_args(self, ap):
        _deploy.add_args(ap)

    def deploy(self, args, ap):
        return _deploy.run(args, ap)

    def list(self):
        _gen.list_scenarios()
