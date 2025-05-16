from dataclasses import dataclass, is_dataclass


# decorator to wrap original __init__
def has_nested_dataclass(*args, **kwargs):
    def wrapper(check_class):

        # passing class to investigate
        check_class = dataclass(check_class, **kwargs)
        o_init = check_class.__init__

        def __init__(self, *args, **kwargs):

            for name, value in kwargs.items():

                # getting field type
                ft = check_class.__annotations__.get(name, None)

                if is_dataclass(ft) and isinstance(value, dict):
                    obj = ft(**value)
                    kwargs[name] = obj
                o_init(self, *args, **kwargs)

        check_class.__init__ = __init__

        return check_class

    return wrapper(args[0]) if args else wrapper


def convert_crlf_to_lf(s):
    """Convert CRLF to LF in a string."""
    if isinstance(s, str):
        return s.replace("\r", "").replace("\r\n", "\n")
    elif isinstance(s, dict):
        # 如果是字典，递归处理所有字符串值
        return {k: convert_crlf_to_lf(v) for k, v in s.items()}
    elif isinstance(s, list):
        # 如果是列表，递归处理所有元素
        return [convert_crlf_to_lf(item) for item in s]
    return s
