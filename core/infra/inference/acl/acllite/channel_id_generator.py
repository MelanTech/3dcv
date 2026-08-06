import threading


class _ChannelIdGenerator(object):
    """Generate global unique id number."""

    _instance_lock = threading.Lock()
    channel_id = 0

    def __new__(cls, *args, **kwargs):
        if not hasattr(_ChannelIdGenerator, "_instance"):
            with _ChannelIdGenerator._instance_lock:
                if not hasattr(_ChannelIdGenerator, "_instance"):
                    _ChannelIdGenerator._instance = object.__new__(cls, *args, **kwargs)
        return _ChannelIdGenerator._instance

    def generator_channel_id(self):
        with _ChannelIdGenerator._instance_lock:
            current_channel_id = _ChannelIdGenerator.channel_id
            _ChannelIdGenerator.channel_id += 1
        return current_channel_id
