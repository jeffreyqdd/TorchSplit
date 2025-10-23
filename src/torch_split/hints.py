import enum 

class HintType(enum.Enum):
        



def with_hint(x):
    setattr(x, "_dynamo_hint_lmao", 12345)
    return x


def compiler_hint(hint_value):
    def wrap(fn):
        fn._compiler_hint = hint_value
        return fn

    return wrap
